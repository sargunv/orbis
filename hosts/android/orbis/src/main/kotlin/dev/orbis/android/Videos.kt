package dev.orbis.android

import android.graphics.ImageFormat
import android.media.Image
import android.media.ImageReader
import android.media.MediaPlayer
import android.os.Handler
import android.os.HandlerThread
import dev.orbis.OrbErrorException
import dev.orbis.OrbEvent
import dev.orbis.OrbEventKind
import dev.orbis.OrbMap

/**
 * Video sources' players: `MediaPlayer` (MediaCodec
 * underneath) plays each video the map asks for into an [ImageReader] on a
 * video thread, and each frame goes to the map as RGBA rows on the main
 * thread. Android reads MP4 and WebM (H.264, HEVC, VP8, VP9).
 */
internal class Videos(
    private val map: OrbMap,
    /** Runs on the main thread. */
    private val main: (Runnable) -> Unit,
    /** A frame arrived: draw. */
    private val changed: () -> Unit,
) : AutoCloseable {
    private val thread = HandlerThread("OrbisVideo").apply { start() }
    private val handler = Handler(thread.looper)
    /** Players by source, touched on the video thread only. */
    private val players = HashMap<String, Player>()
    /** Closed, on the main thread: frames still on their way are dropped. */
    private var closed = false

    /** Runs `call` with the map on the main thread unless closed; a source
     * gone meanwhile is fine, its close is on the way. */
    private fun withMap(call: (OrbMap) -> Unit) = main(Runnable {
        if (!closed) {
            try {
                call(map)
            } catch (e: OrbErrorException) {
                // The source is gone.
            }
        }
    })

    /** Acts on a video event, on the main thread; false for other events. */
    fun handle(event: OrbEvent): Boolean {
        val source = event.text()
        when (event.kind()) {
            OrbEventKind.VideoOpen -> {
                val urls = event.urls().split("\n")
                handler.post {
                    players.remove(source)?.close()
                    players[source] = Player(source, urls)
                }
            }
            OrbEventKind.VideoPlay -> handler.post { players[source]?.player?.start() }
            OrbEventKind.VideoPause -> handler.post { players[source]?.player?.pause() }
            OrbEventKind.VideoSeek -> {
                val ms = (event.seconds() * 1000).toInt()
                handler.post { players[source]?.player?.seekTo(ms) }
            }
            OrbEventKind.VideoClose -> handler.post { players.remove(source)?.close() }
            else -> return false
        }
        return true
    }

    override fun close() {
        closed = true
        handler.post {
            players.values.forEach { it.close() }
            players.clear()
            thread.quitSafely()
        }
    }

    private inner class Player(val source: String, urls: List<String>) {
        var player: MediaPlayer? = null
        private var reader: ImageReader? = null

        init {
            // The first URL that prepares plays (GL JS `getVideo`).
            for (url in urls) {
                val candidate = MediaPlayer()
                try {
                    candidate.setDataSource(url)
                    candidate.prepare()
                } catch (e: Exception) {
                    candidate.release()
                    continue
                }
                val reader = ImageReader.newInstance(
                    candidate.videoWidth, candidate.videoHeight, ImageFormat.YUV_420_888, 3,
                )
                reader.setOnImageAvailableListener({ r ->
                    r.acquireLatestImage()?.use { deliver(it) }
                }, handler)
                candidate.setSurface(reader.surface)
                candidate.isLooping = true
                candidate.setVolume(0f, 0f)
                candidate.start()
                this.player = candidate
                this.reader = reader
                break
            }
            if (player == null) withMap { it.videoFailed(source, "Android plays none of $urls") }
        }

        private fun deliver(image: Image) {
            val (width, height) = image.width to image.height
            val rgba = rgba(image)
            withMap {
                it.sourceFrame(source, width.toUInt(), height.toUInt(), rgba)
                changed()
            }
        }

        fun close() {
            player?.release()
            reader?.close()
        }
    }

    private companion object {
        /** A YUV 4:2:0 image as RGBA rows (BT.601, limited range). */
        fun rgba(image: Image): ByteArray {
            val (width, height) = image.width to image.height
            val (y, u, v) = image.planes
            val out = ByteArray(width * height * 4)
            for (row in 0 until height) {
                for (col in 0 until width) {
                    val luma = (y.buffer.get(row * y.rowStride + col * y.pixelStride).toInt() and 0xff) - 16
                    val c = (row / 2) * u.rowStride + (col / 2) * u.pixelStride
                    val d = (row / 2) * v.rowStride + (col / 2) * v.pixelStride
                    val cb = (u.buffer.get(c).toInt() and 0xff) - 128
                    val cr = (v.buffer.get(d).toInt() and 0xff) - 128
                    val l = 298 * luma
                    val at = (row * width + col) * 4
                    out[at] = ((l + 409 * cr + 128) shr 8).coerceIn(0, 255).toByte()
                    out[at + 1] = ((l - 100 * cb - 208 * cr + 128) shr 8).coerceIn(0, 255).toByte()
                    out[at + 2] = ((l + 516 * cb + 128) shr 8).coerceIn(0, 255).toByte()
                    out[at + 3] = -1
                }
            }
            return out
        }
    }
}
