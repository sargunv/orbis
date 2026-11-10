package dev.orbis.android

import android.annotation.SuppressLint
import android.content.Context
import android.view.Choreographer
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import dev.orbis.OrbButton
import dev.orbis.OrbError
import dev.orbis.OrbErrorException
import dev.orbis.OrbEvent
import dev.orbis.OrbFrame
import dev.orbis.OrbMap
import dev.orbis.OrbModifiers
import dev.orbis.OrbPointerKind
import dev.orbis.OrbRuntime
import dev.orbis.OrbTarget

/**
 * A view that draws an Orbis map into its surface (Vulkan), paced by Choreographer, and hands touch
 * and mouse input to the map's gestures with their event times (`MotionEvent` times and historical
 * samples, on the clock Choreographer's frame times use). It draws only while the map needs frames:
 * an idle map costs no frames, and data arriving on Orbis's threads wakes it. The map outlives the
 * surface: leaving the screen drops the target and keeps the map. A lost GPU is replaced, and the
 * map drawn again on the new one.
 *
 * Use it on the main thread; [close] it with the runtime's other objects.
 */
@SuppressLint("ViewConstructor")
class MapView(context: Context, val runtime: OrbRuntime) :
  SurfaceView(context), SurfaceHolder.Callback, Choreographer.FrameCallback, AutoCloseable {

  val map: OrbMap = OrbMap.create(runtime, 1.0, 1.0, 1.0, 1.0)

  /** Called after each frame drawn, with what it did. */
  var onFrame: ((OrbFrame) -> Unit)? = null

  /** Called with each event; the event is closed after the call. */
  var onEvent: ((OrbEvent) -> Unit)? = null

  /** Frames drawn since the view was made. */
  var framesDrawn = 0
    private set

  private val density = resources.displayMetrics.density.toDouble()
  private var target: OrbTarget? = null
  private var window = 0L
  private var scheduled = false
  /** The style's video sources, played with MediaPlayer. */
  private val videos = Videos(map, { post(it) }, { setNeedsFrame() })

  init {
    holder.addCallback(this)
    map.setPlaysVideo(true)
    map.setWake { post { setNeedsFrame() } }
  }

  /** Draws frames until the map needs none; call after changing the map. */
  fun setNeedsFrame() {
    if (!scheduled && target != null) {
      scheduled = true
      Choreographer.getInstance().postFrameCallback(this)
    }
  }

  override fun surfaceCreated(holder: SurfaceHolder) {}

  override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
    val current = target
    if (current != null) {
      current.resize(width.toUInt(), height.toUInt(), density)
    } else {
      window = NativeWindows.fromSurface(holder.surface)
      target =
        OrbTarget.createAndroidWindow(
          runtime,
          window.toULong(),
          width.toUInt(),
          height.toUInt(),
          density,
        )
    }
    setNeedsFrame()
  }

  override fun surfaceDestroyed(holder: SurfaceHolder) {
    dropTarget()
  }

  private fun dropTarget() {
    target?.close()
    target = null
    if (window != 0L) {
      NativeWindows.release(window)
      window = 0L
    }
  }

  override fun doFrame(frameTimeNanos: Long) {
    scheduled = false
    val target = target ?: return
    try {
      val frame = map.render(target, frameTimeNanos / 1e6)
      if (frame.drawn) {
        framesDrawn++
        onFrame?.invoke(frame)
      }
    } catch (e: OrbErrorException) {
      when (e.error) {
        // A new GPU, or the surface went: a surface on it.
        OrbError.TargetLost -> {
          dropTarget()
          holder.surface?.let { surfaceChanged(holder, 0, width, height) }
        }
        else -> throw e
      }
    }
    while (true) {
      val event = map.pollEvent() ?: break
      event.use {
        videos.handle(it)
        onEvent?.invoke(it)
      }
    }
    if (map.needsFrame()) setNeedsFrame()
  }

  @SuppressLint("ClickableViewAccessibility")
  override fun onTouchEvent(event: MotionEvent): Boolean {
    val kind =
      when (event.getToolType(event.actionIndex)) {
        MotionEvent.TOOL_TYPE_MOUSE -> OrbPointerKind.Mouse
        MotionEvent.TOOL_TYPE_STYLUS -> OrbPointerKind.Pen
        else -> OrbPointerKind.Touch
      }
    val index = event.actionIndex
    val id = event.getPointerId(index).toUInt()
    val x = event.getX(index) / density
    val y = event.getY(index) / density
    // Milliseconds of uptime: Choreographer's frame times' clock.
    val time = event.eventTime.toDouble()
    when (event.actionMasked) {
      MotionEvent.ACTION_DOWN,
      MotionEvent.ACTION_POINTER_DOWN ->
        map.pointerDown(id, kind, OrbButton.Primary, x, y, NO_MODIFIERS, time)
      MotionEvent.ACTION_MOVE -> {
        // The samples batched since the last event, then its own.
        for (h in 0 until event.historySize) {
          val at = event.getHistoricalEventTime(h).toDouble()
          for (i in 0 until event.pointerCount) {
            map.pointerMove(
              event.getPointerId(i).toUInt(),
              event.getHistoricalX(i, h) / density,
              event.getHistoricalY(i, h) / density,
              at,
            )
          }
        }
        for (i in 0 until event.pointerCount) {
          map.pointerMove(
            event.getPointerId(i).toUInt(),
            event.getX(i) / density,
            event.getY(i) / density,
            time,
          )
        }
      }
      MotionEvent.ACTION_UP,
      MotionEvent.ACTION_POINTER_UP -> map.pointerUp(id, x, y, time)
      MotionEvent.ACTION_CANCEL ->
        for (i in 0 until event.pointerCount) {
          map.pointerCancel(event.getPointerId(i).toUInt(), time)
        }
      else -> return super.onTouchEvent(event)
    }
    setNeedsFrame()
    return true
  }

  override fun close() {
    holder.removeCallback(this)
    Choreographer.getInstance().removeFrameCallback(this)
    dropTarget()
    videos.close()
    map.close()
  }

  private companion object {
    val NO_MODIFIERS = OrbModifiers(false, false, false, false)
  }
}
