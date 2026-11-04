// Calls every shape of the fixture bridge through the generated Swift.
import Foundation

// Releasing an owner-thread object on another thread must stop the
// program, not destroy it under its owner: run.rs expects this to trap.
if CommandLine.arguments.contains("--release-elsewhere") {
    final class Box: @unchecked Sendable {
        var counter: Counter?
    }
    let box = Box()
    box.counter = Counter.create(start: 0)
    let thread = Thread { box.counter = nil }
    thread.start()
    Thread.sleep(forTimeInterval: 5)
    print("released elsewhere without trapping")
    exit(0)
}

func expectThrows<E: Error>(_ type: E.Type, _ body: () throws -> Void) -> E {
    do {
        try body()
    } catch let error as E {
        return error
    } catch {
        fatalError("threw \(error), not \(E.self)")
    }
    fatalError("didn't throw")
}

let counter = Counter.create(start: 1)
precondition(counter.add(by: 2) == 3)
precondition(counter.get() == 3)

let empty = expectThrows(FixtureErrorFailure.self) { try counter.setName(name: "") }
precondition(empty.error == .bad && empty.message == "the name is empty", "\(empty)")
let noName = expectThrows(FixtureErrorFailure.self) { _ = try counter.checkedName() }
precondition(noName.error == .worse && noName.message == "no name")
try! counter.setName(name: "héllo 🌍")
precondition(counter.name() == "héllo 🌍")
precondition(try! counter.checkedName() == "héllo 🌍")

precondition(counter.sum(values: [1.5, 2.5]) == 4.0)
precondition(counter.sum(values: []) == 0.0)
precondition(counter.bytes() == [1, 2, 255])

let point = Sample(x: 1.5, y: -3, big: UInt64.max, size: 7, flag: true, inner: Inner(a: 200, color: .green))
precondition(counter.echo(point: point) == point)

precondition(try! counter.divide(by: 3) == 1)
let zero = expectThrows(FixtureErrorFailure.self) { _ = try counter.divide(by: 0) }
precondition(zero.message == "division by zero")
precondition(try! counter.checked(ok: true) == 7)
_ = expectThrows(DiplomatCallFailure.self) { _ = try counter.checked(ok: false) }

let item = counter.item()
precondition(item.value() == 3)
precondition(counter.find(present: false) == nil)
precondition(counter.find(present: true)!.value() == 42)
precondition(counter.absorb(item: item) == 6)
precondition(counter.color() == .red)
precondition(FixtureError.allCases.map(\.rawValue) == [1, 2])

final class Calls: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var threads = Set<String>()
    func hit() {
        lock.lock()
        count += 1
        threads.insert("\(pthread_self())")
        lock.unlock()
    }
    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }
}
let calls = Calls()
counter.setWake { calls.hit() }
counter.fire(times: 3)
precondition(calls.value == 3)
precondition(!calls.threads.contains("\(pthread_self())"), "called on another thread")
counter.setWake {}
counter.fire(times: 1)
precondition(calls.value == 3)

// A trivial call through the binding.
let n = 20_000_000
var sink: Int64 = 0
let started = DispatchTime.now().uptimeNanoseconds
for _ in 0..<n {
    sink &+= counter.get()
}
let elapsed = DispatchTime.now().uptimeNanoseconds - started
print(String(format: "swift: get() %.2f ns/call (%lld)", Double(elapsed) / Double(n), sink))
print("swift ok")
