//! Opening connections: ureq's connector chain with its TCP step racing a
//! host's addresses (Happy Eyeballs, RFC 8305), as curl and browsers do.
//! ureq alone tries each address in turn, waiting out the first.

use std::net::SocketAddr;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use ureq::unversioned::resolver::{DefaultResolver, Resolver};
use ureq::unversioned::transport::{
    ConnectProxyConnector, ConnectionDetails, Connector, Either, RustlsConnector, TcpConnector,
    Transport,
};

/// How long an attempt runs alone before the next address joins the race:
/// curl's `CURLOPT_HAPPY_EYEBALLS_TIMEOUT_MS`.
const HEAD_START: Duration = Duration::from_millis(200);

/// An agent whose connections race their host's addresses and that keeps
/// up to `idle_per_host` connections to a host open between requests.
pub(crate) fn agent(idle_per_host: usize) -> ureq::Agent {
    agent_with(
        TcpConnector::default(),
        DefaultResolver::default(),
        idle_per_host,
    )
}

fn agent_with(
    per_address: impl Connector,
    resolver: impl Resolver,
    idle_per_host: usize,
) -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .user_agent(concat!("orbis/", env!("CARGO_PKG_VERSION")))
        .max_idle_connections(idle_per_host.max(10))
        .max_idle_connections_per_host(idle_per_host)
        .build();
    // ureq's default chain, with the race for its TCP step.
    let connector = ()
        .chain(ConnectProxyConnector::default())
        .chain(Race {
            per_address: Arc::new(per_address),
        })
        .chain(RustlsConnector::default());
    ureq::Agent::with_parts(config, connector, resolver)
}

/// The TCP step: the addresses alternate between the families, starting
/// with the resolver's first, and each attempt runs [`HEAD_START`] alone
/// (or until it fails) before the next starts beside it. The first to
/// connect wins; the others close as they finish.
#[derive(Debug)]
struct Race<A> {
    per_address: Arc<A>,
}

type Attempt = Result<Box<dyn Transport>, ureq::Error>;

impl<In: Transport, A: Connector> Connector<In> for Race<A> {
    type Out = Either<In, Box<dyn Transport>>;

    fn connect(
        &self,
        details: &ConnectionDetails,
        chained: Option<In>,
    ) -> Result<Option<Self::Out>, ureq::Error> {
        // A CONNECT proxy's tunnel.
        if let Some(chained) = chained {
            return Ok(Some(Either::A(chained)));
        }
        if details.addrs.len() < 2 {
            let transport = self.per_address.connect(details, None)?;
            return Ok(transport.map(|t| Either::B(Box::new(t) as Box<dyn Transport>)));
        }
        let ipv6 = details.addrs[0].is_ipv6();
        let (first, other): (Vec<SocketAddr>, Vec<SocketAddr>) =
            details.addrs.iter().partition(|a| a.is_ipv6() == ipv6);
        let addrs: Vec<SocketAddr> = (0..first.len().max(other.len()))
            .flat_map(|i| first.get(i).into_iter().chain(other.get(i)).copied())
            .collect();
        let (send, receive) = mpsc::channel::<Attempt>();
        let (mut started, mut running, mut last_error) = (0, 0, None);
        loop {
            if started < addrs.len() {
                attempt(&self.per_address, details, addrs[started], send.clone())?;
                started += 1;
                running += 1;
            }
            let result = if started < addrs.len() {
                match receive.recv_timeout(HEAD_START) {
                    Ok(result) => result,
                    // The next address's turn.
                    Err(_) => continue,
                }
            } else if running > 0 {
                receive.recv().expect("a sender is held here")
            } else {
                return Err(last_error.unwrap_or(ureq::Error::ConnectionFailed));
            };
            running -= 1;
            match result {
                Ok(transport) => return Ok(Some(Either::B(transport))),
                Err(error) => last_error = Some(error),
            }
        }
    }
}

/// Connects to `addr` alone on a thread of its own, and sends the outcome.
fn attempt<A: Connector>(
    per_address: &Arc<A>,
    details: &ConnectionDetails,
    addr: SocketAddr,
    send: mpsc::Sender<Attempt>,
) -> Result<(), ureq::Error> {
    let (uri, config) = (details.uri.clone(), details.config.clone());
    let (request_level, now, timeout) = (details.request_level, details.now, details.timeout);
    let current_time = Arc::clone(&details.current_time);
    let run_connector = Arc::clone(&details.run_connector);
    let register_shutdown = Arc::clone(&details.register_shutdown);
    let per_address = Arc::clone(per_address);
    std::thread::Builder::new()
        .name("connect".into())
        .spawn(move || {
            // The TCP step resolves nothing, but the details need a resolver.
            let resolver = DefaultResolver::default();
            let mut addrs = resolver.empty();
            addrs.push(addr);
            let details = ConnectionDetails {
                uri: &uri,
                addrs,
                config: &config,
                request_level,
                resolver: &resolver,
                now,
                timeout,
                current_time,
                run_connector,
                register_shutdown,
            };
            let result = per_address
                .connect(&details, None)
                .and_then(|t| t.ok_or(ureq::Error::ConnectionFailed))
                .map(|t| Box::new(t) as Box<dyn Transport>);
            // Once another attempt has won, nobody listens and this
            // connection closes.
            let _ = send.send(result);
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use ureq::config::Config;
    use ureq::http::Uri;
    use ureq::unversioned::resolver::ResolvedSocketAddrs;
    use ureq::unversioned::transport::NextTimeout;

    /// Resolves every name to an address that never answers a connect,
    /// then to `answers`.
    #[derive(Debug)]
    struct Resolve {
        silent: SocketAddr,
        answers: SocketAddr,
    }

    impl Resolver for Resolve {
        fn resolve(
            &self,
            _: &Uri,
            _: &Config,
            _: NextTimeout,
        ) -> Result<ResolvedSocketAddrs, ureq::Error> {
            let mut addrs = self.empty();
            addrs.push(self.silent);
            addrs.push(self.answers);
            Ok(addrs)
        }
    }

    /// TCP, except that connects to `silent` hang, as to an address whose
    /// packets are dropped.
    #[derive(Debug)]
    struct Hang {
        silent: SocketAddr,
    }

    impl<In: Transport> Connector<In> for Hang {
        type Out = Either<In, Box<dyn Transport>>;

        fn connect(
            &self,
            details: &ConnectionDetails,
            chained: Option<In>,
        ) -> Result<Option<Self::Out>, ureq::Error> {
            if details.addrs[0] == self.silent {
                std::thread::sleep(Duration::from_secs(30));
                return Err(ureq::Error::ConnectionFailed);
            }
            let tcp = TcpConnector::default().connect(details, chained)?;
            Ok(tcp.map(|t| Either::B(Box::new(t) as Box<dyn Transport>)))
        }
    }

    #[test]
    fn a_silent_first_address_costs_a_head_start_not_a_timeout() {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let answers = server.local_addr().unwrap();
        let thread = std::thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let mut request = [0; 1024];
            let _ = stream.read(&mut request).unwrap();
            let reply = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";
            stream.write_all(reply.as_bytes()).unwrap();
        });
        // An IPv6 address first, as resolvers order them.
        let silent: SocketAddr = format!("[::1]:{}", answers.port()).parse().unwrap();
        let agent = agent_with(Hang { silent }, Resolve { silent, answers }, 4);
        let started = std::time::Instant::now();
        let body = agent
            .get("http://tiles.test/")
            .call()
            .unwrap()
            .body_mut()
            .read_to_string()
            .unwrap();
        assert_eq!(body, "ok");
        let took = started.elapsed();
        assert!(
            took >= HEAD_START && took < Duration::from_secs(5),
            "{took:?}"
        );
        thread.join().unwrap();
    }
}
