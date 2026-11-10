// Calls every shape of the fixture bridge through the generated Kotlin.
import dev.fixture.*
import java.util.concurrent.atomic.AtomicInteger

inline fun <reified E : Throwable> expectThrows(body: () -> Unit): E {
  try {
    body()
  } catch (e: Throwable) {
    if (e is E) return e
    throw AssertionError("threw $e, not ${E::class}")
  }
  throw AssertionError("didn't throw")
}

fun main() {
  Counter.create(1).use { counter ->
    check(counter.add(2) == 3L)
    check(counter.get() == 3L)

    val empty = expectThrows<FixtureErrorException> { counter.setName("") }
    check(empty.error == FixtureError.Bad && empty.message == "the name is empty") { "$empty" }
    val noName = expectThrows<FixtureErrorException> { counter.checkedName() }
    check(noName.error == FixtureError.Worse && noName.message == "no name")
    counter.setName("héllo 🌍")
    check(counter.name() == "héllo 🌍")
    check(counter.checkedName() == "héllo 🌍")

    check(counter.sum(doubleArrayOf(1.5, 2.5)) == 4.0)
    check(counter.sum(doubleArrayOf()) == 0.0)
    check(counter.bytes().contentEquals(byteArrayOf(1, 2, -1)))

    val point = Sample(1.5, -3, ULong.MAX_VALUE, 7u, true, Inner(200u, Color.Green))
    check(counter.echo(point) == point) { "${counter.echo(point)}" }

    check(counter.divide(3) == 1L)
    val zero = expectThrows<FixtureErrorException> { counter.divide(0) }
    check(zero.message == "division by zero")
    check(counter.checked(true) == 7u)
    expectThrows<RuntimeException> { counter.checked(false) }

    counter.item().use { item ->
      check(item.value() == 3L)
      check(counter.find(false) == null)
      counter.find(true)!!.use { check(it.value() == 42L) }
      check(counter.absorb(item) == 6L)
    }
    check(counter.color() == Color.Red)

    val calls = AtomicInteger()
    val caller = Thread.currentThread()
    val threads = java.util.Collections.synchronizedList(mutableListOf<Thread>())
    counter.setWake {
      calls.incrementAndGet()
      threads += Thread.currentThread()
    }
    counter.fire(3u)
    check(calls.get() == 3)
    check(threads.none { it === caller })
    // The library's thread is detached after each call, so each call
    // attaches it afresh: the thread may exit without the JVM's leave.
    check(threads.toSet().size == 3) { "attached once and kept: $threads" }
    counter.setWake {}
    counter.fire(1u)
    check(calls.get() == 3)

    // Closing an object any thread may use waits for a call under way
    // on another thread, rather than freeing it under the call.
    counter.item().let { item ->
      var seen = 0L
      val reader = Thread { seen = item.slowValue(300u) }
      reader.start()
      Thread.sleep(50)
      val closing = System.nanoTime()
      item.close()
      val waited = (System.nanoTime() - closing) / 1_000_000
      reader.join()
      check(seen == 6L) { "read $seen" }
      check(waited >= 150) { "close returned after $waited ms, under the call" }
      check(runCatching { item.value() }.exceptionOrNull() is IllegalStateException)
    }

    // Owner-thread objects refuse other threads.
    var refused = false
    val t = Thread {
      refused = runCatching { counter.get() }.exceptionOrNull() is IllegalStateException
    }
    t.start()
    t.join()
    check(refused)

    // A trivial call through the binding.
    val n = 20_000_000
    var sink = 0L
    repeat(n / 10) { sink += counter.get() }
    val started = System.nanoTime()
    repeat(n) { sink += counter.get() }
    val elapsed = System.nanoTime() - started
    println("kotlin: get() %.2f ns/call (%d)".format(elapsed.toDouble() / n, sink))
  }
  val closed = Counter.create(0)
  closed.close()
  expectThrows<IllegalStateException> { closed.get() }
  println("kotlin ok")
}
