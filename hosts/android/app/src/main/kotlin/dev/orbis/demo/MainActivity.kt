package dev.orbis.demo

import android.app.Activity
import android.graphics.Color
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.widget.FrameLayout
import android.widget.TextView
import android.os.Build
import dev.orbis.OrbBackend
import dev.orbis.OrbCameraTarget
import dev.orbis.OrbEventKind
import dev.orbis.OrbMap
import dev.orbis.OrbRuntime
import dev.orbis.android.MapView
import org.json.JSONArray
import java.io.File

/**
 * The Android reference app: the web demo's style with gestures, a popup
 * from queryRenderedFeatures on tap, hover names with a mouse, and a pin at
 * Tokyo Station that follows the map. Logs load timings (tag "Orbis").
 *
 * The web demo's live OpenFreeMap Liberty over Tokyo by default; the
 * committed extract with `adb -s "$(scripts/android-serial)" shell am start -n
 * dev.orbis.demo/.MainActivity --es style tokyo`. `--es backend compat` (or
 * `modern`) picks the GPU tier instead of the device policy; `--el lose <ms>`
 * loses the GPU that long after loading, to show recovery.
 */
class MainActivity : Activity() {
    private lateinit var runtime: OrbRuntime
    private lateinit var view: MapView
    private lateinit var status: TextView
    private val pins = mutableListOf<Pin>()
    private var popup: Pin? = null
    private var pendingQuery: Triple<ULong, Double, Double>? = null
    private var loadLogged = false
    private var firstDrawLogged = false
    private var compiledLogged = false
    /** When the GPU was lost on purpose, until a complete frame. */
    private var lostAt: Long? = null

    /** A label pinned to a place on the map. */
    private inner class Pin(text: String, val lon: Double, val lat: Double, background: Int) {
        val label = TextView(this@MainActivity).apply {
            this.text = text
            setBackgroundColor(background)
            setTextColor(Color.BLACK)
            setPadding(12, 6, 12, 6)
        }

        fun place(map: OrbMap) {
            val p = map.project(lon, lat)
            val density = resources.displayMetrics.density
            label.visibility = if (p.valid) View.VISIBLE else View.INVISIBLE
            label.x = (p.x * density).toFloat() - label.width / 2f
            label.y = (p.y * density).toFloat() - label.height - 12f
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val started = SystemClock.elapsedRealtime()
        val data = unpackData()
        val backend = when (intent.getStringExtra("backend")) {
            "compat" -> OrbBackend.Compat
            "modern" -> OrbBackend.Modern
            else -> OrbBackend.Auto
        }
        runtime = OrbRuntime.createWith(
            0u, backend, File(noBackupFilesDir, "orbis-state").path, Build.VERSION.SDK_INT.toUInt(),
        )
        Log.i("Orbis", runtime.selection())
        runtime.setPipelineCacheDir(File(cacheDir, "orbis-pipelines").path)
        val deviceMs = SystemClock.elapsedRealtime() - started
        view = MapView(this, runtime)
        status = TextView(this).apply {
            setBackgroundColor(Color.argb(200, 255, 255, 255))
            setTextColor(Color.BLACK)
            textSize = 11f
            setPadding(12, 6, 12, 6)
        }
        val root = FrameLayout(this)
        root.addView(view)
        root.addView(status, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.START))
        setContentView(root)
        addPin(Pin("Tokyo Station", 139.7671, 35.6812, Color.YELLOW))

        if (intent.getStringExtra("style") == "tokyo") {
            val style = File(data, "tokyo-mvt.json").readText()
                .replace("pmtiles://./data/", "pmtiles://file://${data.path}/")
                .replace("\"./data/", "\"file://${data.path}/")
            view.map.setStyleJson(style)
        } else {
            view.map.jumpTo(OrbCameraTarget(139.767, 35.681, 13.0, Double.NaN, Double.NaN, Double.NaN, Double.NaN))
            view.map.loadStyleUrl("https://tiles.openfreemap.org/styles/liberty")
        }

        view.onFrame = { frame ->
            for (pin in pins + listOfNotNull(popup)) pin.place(view.map)
            val c = view.map.camera()
            status.text = "z%.2f · %d tiles · %d draws · %s · frames %d%s".format(
                c.zoom, frame.tilesDrawn.toInt(), frame.draws.toInt(),
                if (frame.idle) "idle" else "loading", view.framesDrawn,
                if (frame.pipelinesPending > 0u) " · %d pipelines compiling".format(frame.pipelinesPending.toInt()) else "",
            )
            // Load to the first frame that draws a layer, and to the last
            // pipeline (plan §18's pipeline-load gate).
            if (!firstDrawLogged && frame.tilesDrawn > 0u && frame.draws > 0u) {
                firstDrawLogged = true
                Log.i("Orbis", "first draw %d ms after start; %d pipelines compiling, %d layers waiting".format(
                    SystemClock.elapsedRealtime() - started, frame.pipelinesPending.toInt(), frame.layersWaiting.toInt()))
            }
            if (!compiledLogged && frame.tilesDrawn > 0u && frame.pipelinesPending == 0u) {
                compiledLogged = true
                Log.i("Orbis", "pipelines compiled %d ms after start".format(SystemClock.elapsedRealtime() - started))
            }
            lostAt?.let { at ->
                if (frame.idle) {
                    lostAt = null
                    Log.i("Orbis", "recovered from device loss: complete frame %d ms after it".format(
                        SystemClock.elapsedRealtime() - at))
                }
            }
            if (frame.idle && !loadLogged) {
                loadLogged = true
                val lose = intent.getLongExtra("lose", -1)
                if (lose >= 0) {
                    view.postDelayed({
                        lostAt = SystemClock.elapsedRealtime()
                        runtime.loseDevice()
                        view.setNeedsFrame()
                    }, lose)
                }
                Log.i(
                    "Orbis",
                    "loaded in %d ms (device %d ms) after %d frames; %d pipelines in %.0f ms; %d tiles, %d draws".format(
                        SystemClock.elapsedRealtime() - started, deviceMs, view.framesDrawn,
                        runtime.pipelinesCreated().toInt(), runtime.pipelineMs(),
                        frame.tilesDrawn.toInt(), frame.draws.toInt(),
                    ),
                )
                for (line in runtime.pipelineTimings().lines().filter { it.isNotBlank() }) {
                    Log.i("Orbis", "  pipeline %s ms".format(line))
                }
            }
        }
        view.onRecovered = { ms -> Log.i("Orbis", "new device in %d ms: %s".format(ms, runtime.selection())) }
        view.onEvent = { event ->
            when (event.kind()) {
                OrbEventKind.Click -> {
                    Log.i("Orbis", "tap at %.0f, %.0f".format(event.x(), event.y()))
                    val at = view.map.unproject(event.x(), event.y())
                    val id = view.map.queryRendered(event.x(), event.y(), event.x(), event.y())
                    pendingQuery = Triple(id, at.lon, at.lat)
                }
                OrbEventKind.QueryResult -> if (event.id() == pendingQuery?.first) showPopup(event.text())
                OrbEventKind.Error -> Log.w("Orbis", event.text())
                else -> {}
            }
        }
        view.setOnGenericMotionListener { _, e ->
            if (e.actionMasked == MotionEvent.ACTION_HOVER_MOVE) {
                val density = resources.displayMetrics.density
                view.map.hitTest(e.x / density.toDouble(), e.y / density.toDouble())?.use { hit ->
                    status.text = "${hit.layer()}${if (hit.hasId()) " #${hit.id()}" else ""}"
                }
                true
            } else {
                false
            }
        }
    }

    private fun addPin(pin: Pin) {
        pins += pin
        (view.parent as FrameLayout).addView(pin.label, FrameLayout.LayoutParams(-2, -2))
    }

    /** A popup with the topmost feature's layer and properties. */
    private fun showPopup(json: String) {
        popup?.let { (view.parent as FrameLayout).removeView(it.label) }
        popup = null
        val (_, lon, lat) = pendingQuery ?: return
        val features = JSONArray(json)
        if (features.length() == 0) {
            Log.i("Orbis", "tapped nothing")
            return
        }
        val feature = features.getJSONObject(0)
        val layer = feature.optJSONObject("layer")?.optString("id") ?: "?"
        val properties = feature.optJSONObject("properties")
        val lines = mutableListOf(layer)
        properties?.keys()?.asSequence()?.sorted()?.take(6)?.forEach { lines += "$it: ${properties.get(it)}" }
        val pin = Pin(lines.joinToString("\n"), lon, lat, Color.WHITE)
        popup = pin
        (view.parent as FrameLayout).addView(pin.label, FrameLayout.LayoutParams(-2, -2))
        Log.i("Orbis", "tapped ${lines.joinToString(", ")}")
        view.setNeedsFrame()
    }

    /** Copies the bundled data to files once per app version. */
    private fun unpackData(): File {
        val out = File(filesDir, "data-${packageManager.getPackageInfo(packageName, 0).lastUpdateTime}")
        if (!out.isDirectory) {
            filesDir.listFiles()?.filter { it.name.startsWith("data-") }?.forEach { it.deleteRecursively() }
            copyAssets("", out)
        }
        return out
    }

    private fun copyAssets(path: String, to: File) {
        val children = assets.list(path).orEmpty()
        if (children.isEmpty()) {
            to.parentFile?.mkdirs()
            assets.open(path).use { input -> to.outputStream().use { input.copyTo(it) } }
            return
        }
        for (child in children) {
            if (path.isEmpty() && child in setOf("images", "webkit", "geoid_map")) continue
            copyAssets(if (path.isEmpty()) child else "$path/$child", File(to, child))
        }
    }

    override fun onDestroy() {
        view.close()
        runtime.close()
        super.onDestroy()
    }
}
