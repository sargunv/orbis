//! Opening connections: ureq's connector chain with its TCP step racing a
//! host's addresses (Happy Eyeballs, RFC 8305), as curl and browsers do.
//! ureq alone tries each address in turn, waiting out the first.
//! HTTPS runs over rustls with ring and verifies servers as the platform
//! does (rustls-platform-verifier): its trust store, enterprise and user
//! roots, and revocation, rather than a copy of Mozilla's roots. On Android
//! (a `standalone` build with no JVM) it trusts the system store's
//! certificate files instead.

use std::net::SocketAddr;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use orb_host::FetchError;
use ureq::tls::{RootCerts, TlsConfig};
use ureq::unversioned::resolver::{DefaultResolver, Resolver};
use ureq::unversioned::transport::{
    ConnectProxyConnector, ConnectionDetails, Connector, Either, RustlsConnector, TcpConnector,
    Transport,
};

/// How long an attempt runs alone before the next address joins the race:
/// curl's `CURLOPT_HAPPY_EYEBALLS_TIMEOUT_MS`.
const HEAD_START: Duration = Duration::from_millis(200);

/// Android's system certificate store: Conscrypt's, which Android 14 and
/// later update, then the system image's. Neither has the certificates a
/// user installed.
#[cfg(target_os = "android")]
const SYSTEM_STORES: [&str; 2] = [
    "/apex/com.android.conscrypt/cacerts",
    "/system/etc/security/cacerts",
];

/// An agent whose connections race their host's addresses and that keeps
/// up to `idle_per_host` connections to a host open between requests. Its
/// HTTPS verifies servers as the platform does, or on Android with the
/// system store's certificates.
pub(crate) fn agent(idle_per_host: usize) -> ureq::Agent {
    #[cfg(target_os = "android")]
    let roots = certificates_in(&SYSTEM_STORES.map(std::path::Path::new));
    #[cfg(not(target_os = "android"))]
    let roots = RootCerts::PlatformVerifier;
    agent_with(
        TcpConnector::default(),
        DefaultResolver::default(),
        (idle_per_host, roots),
    )
}

/// GETs `url` with `headers`, following its redirects if `follow` says.
pub(crate) fn get(
    agent: &ureq::Agent,
    url: &str,
    headers: &[(&str, &str)],
    follow: bool,
) -> Result<Box<dyn crate::Reply>, FetchError> {
    // A failed response's headers say when to ask again.
    let mut request = agent.get(url).config().http_status_as_error(false);
    if !follow {
        request = request.max_redirects(0);
    }
    let mut request = request.build();
    for &(name, value) in headers {
        request = request.header(name, value);
    }
    let response = request
        .call()
        .map_err(|e| FetchError::Network(e.to_string()))?;
    Ok(Box::new(response))
}

impl crate::Reply for ureq::http::Response<ureq::Body> {
    fn status(&self) -> u16 {
        ureq::http::Response::status(self).as_u16()
    }

    fn header(&self, name: &str) -> Option<String> {
        self.headers()
            .get(name)
            .and_then(|h| h.to_str().ok())
            .map(str::to_owned)
    }

    fn body(
        mut self: Box<Self>,
        limit: u64,
        abort: &orb_host::Abort,
    ) -> Result<Vec<u8>, FetchError> {
        let length = ureq::http::Response::body(&self).content_length();
        crate::read_body(self.body_mut().as_reader(), length, limit, abort)
    }
}

fn agent_with(
    per_address: impl Connector,
    resolver: impl Resolver,
    (idle_per_host, roots): (usize, RootCerts),
) -> ureq::Agent {
    let tls = TlsConfig::builder()
        .root_certs(roots)
        .unversioned_rustls_crypto_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .build();
    let config = ureq::Agent::config_builder()
        .user_agent(crate::USER_AGENT)
        .max_idle_connections(idle_per_host.max(10))
        .max_idle_connections_per_host(idle_per_host)
        .tls_config(tls)
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

/// The PEM certificates in the first of `dirs` holding any (Android's
/// system store, one file a certificate), or none.
#[cfg(any(target_os = "android", test))]
pub(crate) fn certificates_in(dirs: &[&std::path::Path]) -> RootCerts {
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        let certificates: Vec<_> = entries
            .filter_map(|entry| std::fs::read(entry.ok()?.path()).ok())
            .filter_map(|pem| ureq::tls::Certificate::from_pem(&pem).ok())
            .collect();
        if !certificates.is_empty() {
            return RootCerts::new_with_certs(&certificates);
        }
    }
    RootCerts::new_with_certs(&[])
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
        let agent = agent_with(
            Hang { silent },
            Resolve { silent, answers },
            (4, RootCerts::PlatformVerifier),
        );
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

    /// HTTPS verifies servers as the platform does, not with a copy of
    /// Mozilla's roots (#188).
    #[cfg(not(target_os = "android"))]
    #[test]
    fn https_verifies_with_the_platform() {
        let roots = agent(4).config().tls_config().root_certs().clone();
        assert!(matches!(roots, RootCerts::PlatformVerifier));
    }

    /// A self-signed certificate, followed by text as in Android's store
    /// files.
    const CERTIFICATE: &str = "-----BEGIN CERTIFICATE-----
MIICEDCCAbYCCQDTvof//xWPnzAKBggqhkjOPQQDAjAVMRMwEQYDVQQDDApvcmJp
cy10ZXN0MCAXDTI2MTExNDAxMjAwM1oYDzIxMjYxMDIxMDEyMDAzWjAVMRMwEQYD
VQQDDApvcmJpcy10ZXN0MIIBSzCCAQMGByqGSM49AgEwgfcCAQEwLAYHKoZIzj0B
AQIhAP////8AAAABAAAAAAAAAAAAAAAA////////////////MFsEIP////8AAAAB
AAAAAAAAAAAAAAAA///////////////8BCBaxjXYqjqT57PrvVV2mIa8ZR0GsMxT
sPY7zjw+J9JgSwMVAMSdNgiG5wSTamZ44ROdJreBn36QBEEEaxfR8uEsQkf4vObl
Y6RA8ncDfYEt6zOg9KE5RdiYwpZP40Li/hp/m47n60p8D54WK84zV2sxXs7LtkBo
N79R9QIhAP////8AAAAA//////////+85vqtpxeehPO5ysL8YyVRAgEBA0IABCTX
yc6DNcPJ/p6butNmm+cRjkv66jB5EbvNmRZ3b2tnlfQUDpYGJoJG8RaTbkYZbVrN
uDY4FWOPlHkcLFr6oWwwCgYIKoZIzj0EAwIDSAAwRQIhAIgWGiUzQNlgO5sBtv+s
N5qHX+fDwKwdRd+tsAM5O4PyAiAugj7ZoNxvLc1kA7AIMgWou5CUqMdjfSoUaf+E
oloFuA==
-----END CERTIFICATE-----
Certificate:
    Data:
        Version: 1 (0x0)
";

    #[test]
    fn certificates_come_from_the_first_store_that_has_any() {
        let count = |roots: RootCerts| match roots {
            RootCerts::Specific(certificates) => certificates.len(),
            _ => panic!("not specific certificates"),
        };
        let scratch = crate::Scratch::new("stores");
        let [missing, empty, store, later] =
            ["missing", "empty", "store", "later"].map(|name| scratch.join(name));
        for dir in [&empty, &store, &later] {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(store.join("0a1b2c3d.0"), CERTIFICATE).unwrap();
        std::fs::write(store.join("README"), "not a certificate").unwrap();
        std::fs::write(later.join("0a1b2c3d.0"), CERTIFICATE).unwrap();
        std::fs::write(later.join("0a1b2c3d.1"), CERTIFICATE).unwrap();
        assert_eq!(
            count(certificates_in(&[&missing, &empty, &store, &later])),
            1
        );
        assert_eq!(count(certificates_in(&[&later, &store])), 2);
        // Without any, nothing is trusted.
        assert_eq!(count(certificates_in(&[&empty, &missing])), 0);
    }
}
