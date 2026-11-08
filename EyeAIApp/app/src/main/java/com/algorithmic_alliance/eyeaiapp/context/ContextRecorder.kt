package com.algorithmic_alliance.eyeaiapp.context

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.algorithmic_alliance.eyeaiapp.worldstate.FrameStamp
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.*
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

// Explicit local recording only. Inference never waits for filesystem writes.
class ContextRecorder(context: Context): AutoCloseable {
    data class State(val active: Boolean=false,val directory: String?=null,val bytes: Long=0,val dropped: Long=0,val error: String?=null)
    private val context=context.applicationContext
    private val root=File(context.filesDir,"context-diagnostics")
    private val io=Executors.newSingleThreadExecutor()
    private val tasks=AtomicLong()
    private val queued=AtomicLong()
    private val dropped=AtomicLong()
    private val mutable=MutableStateFlow(State())
    val state=mutable.asStateFlow()
    @Volatile private var current: File?=null
    @Volatile private var started=0L
    @Volatile private var lastImage=0L
    private var size=0L
    private val json=Json {encodeDefaults=true
    explicitNulls=true}
    @Synchronized fun start() {
        if(current!=null)return
        if(tasks.get()>0) {mutable.value=mutable.value.copy(error="Vorherige Aufnahme wird noch geschrieben")
        return}
        root.mkdirs()
        val dir=File(root,"${System.currentTimeMillis()}-${UUID.randomUUID().toString().take(8)}")
        dir.mkdirs()
        size=0
dropped.set(0)
started=System.nanoTime()
        current=dir
        mutable.value=State(true,dir.absolutePath)
        event("RECORDING_STARTED",buildJsonObject {
            put("schema","eyeai-context-recording-1")
put("app_git_commit",com.algorithmic_alliance.eyeaiapp.BuildConfig.GIT_COMMIT)
put("app_build_time",com.algorithmic_alliance.eyeaiapp.BuildConfig.BUILD_TIME)
            put("float_format","little-endian float32; nonfinite preserved in binary")
            put("window_before_seconds",10)
            put("window_after_seconds",5)
            put("note","Continuous diagnosis includes missed triggers; event windows index the preceding 10 and following 5 seconds.")
        })
    }
    @Synchronized fun stop() {
        val dir=current
current=null
        mutable.value=mutable.value.copy(active=false)
        if(dir!=null)io.execute {runCatching {File(dir,"events.jsonl").appendText(buildJsonObject {
            put("event","RECORDING_STOPPED")
put("atNanos",System.nanoTime())
            put("dropped",dropped.get())
        }.toString()+"\n")}}
    }
    @Synchronized private fun submit(bytes: Long,onDiscard: ()->Unit = {},work: (File)->Unit): Boolean {
        val dir=current ?: return false
        if(System.nanoTime()-started>600_000_000_000L) {stop()
        return false}
        if(queued.addAndGet(bytes)>64L*1024*1024) {queued.addAndGet(-bytes)
dropped.incrementAndGet()
        return false}
        tasks.incrementAndGet()
        io.execute {
            try {
                if(size+bytes>512L*1024*1024) {onDiscard()
stop()
                mutable.value=mutable.value.copy(error="Speicherlimit der Aufnahme erreicht")}
                else {work(dir)
size+=bytes
                mutable.value=mutable.value.copy(bytes=size,dropped=dropped.get())}
            }catch(e:Exception) {stop()
            mutable.value=mutable.value.copy(error=e.message)}
            finally {queued.addAndGet(-bytes)
            tasks.decrementAndGet()}
        }
        return true
    }
    fun event(kind: String,frame: FrameStamp?=null,data: ()->JsonElement) {
        if(current!=null)event(kind,data(),frame)
    }
    fun event(kind: String,data: JsonElement=JsonNull,frame: FrameStamp?=null) {
        if(current==null)return
        val entry=buildJsonObject {put("event",kind)
put("atNanos",System.nanoTime())
put("frame",frame?.let {json.encodeToJsonElement(it)}?:JsonNull)
        put("data",data)}.toString()+"\n"
        submit(entry.length*3L) { File(it,"events.jsonl").appendText(entry) }
    }
    fun observation(value: ContextObservation,scores: ContextScores,reason: String) {
        if(current==null)return
        event("OBSERVATION",buildJsonObject {
            put("observation",json.encodeToJsonElement(value))
put("scores",json.encodeToJsonElement(scores))
            put("reason",reason)
        },value.frame)
    }
    private fun key(stamp: FrameStamp,channel: String)="${stamp.sessionId}-${stamp.frameId}-$channel"
    fun floats(stamp: FrameStamp,channel: String,width: Int,height: Int,values: FloatArray) {
        if(current==null)return
        val payload=ByteBuffer.allocate(values.size*4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach(payload::putFloat)
        val data=payload.array()
        val filename=key(stamp,channel)+".f32"
        submit(data.size.toLong()) { File(it,filename).writeBytes(data) }
        event("FLOAT_PAYLOAD",buildJsonObject {put("channel",channel)
put("file",filename)
put("width",width)
put("height",height)
put("values",values.size)
        put("nonfinite",values.count {!it.isFinite()})},stamp)
    }
    fun ints(stamp: FrameStamp,channel: String,width: Int,height: Int,values: IntArray) {
        if(current==null)return
        val payload=ByteBuffer.allocate(values.size*4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach(payload::putInt)
        val filename=key(stamp,channel)+".i32"
        val data=payload.array()
        submit(data.size.toLong()) {File(it,filename).writeBytes(data)}
        event("INT_PAYLOAD",buildJsonObject {put("channel",channel)
put("file",filename)
put("width",width)
        put("height",height)},stamp)
    }
    fun image(stamp: FrameStamp,bitmap: Bitmap,trigger: Boolean=false) {
        if(current==null)return
        val now=System.nanoTime()
        if(!trigger&&now-lastImage<500_000_000L)return
        lastImage=now
        val scale=if(trigger)1.0 else minOf(1.0,320.0/bitmap.width)
        val scaled=Bitmap.createScaledBitmap(bitmap,maxOf(1,(bitmap.width*scale).toInt()),maxOf(1,(bitmap.height*scale).toInt()),true)
        val copy=scaled.copy(Bitmap.Config.ARGB_8888,false)
        if(scaled!==bitmap)scaled.recycle()
        val estimate=copy.width.toLong()*copy.height*4
        val submitted=submit(estimate,onDiscard={copy.recycle()}) {dir->
            try {val out=ByteArrayOutputStream()
            copy.compress(if(trigger)Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG,80,out)
                File(dir,key(stamp,if(trigger)"trigger-rgb" else "rgb")+if(trigger)".png" else ".jpg").writeBytes(out.toByteArray())
            }finally {copy.recycle()}
        }
        if(!submitted)copy.recycle()
    }
    fun assetMetadata(paths: List<String>) {
        submit(0) {dir->
            val result=buildJsonObject {
                for(path in paths.distinct())put(path,runCatching {
                    val digest=java.security.MessageDigest.getInstance("SHA-256")
                    context.assets.open(path).use {input->val buffer=ByteArray(32768)
while(true) {val n=input.read(buffer)
if(n<0)break
                    digest.update(buffer,0,n)}}
                    buildJsonObject {put("sha256",digest.digest().joinToString("") {"%02x".format(it.toInt() and 255)})
                        if(path.endsWith(".json"))put("content",Json.parseToJsonElement(context.assets.open(path).bufferedReader().use {it.readText()}))}
                }.getOrElse {buildJsonObject {put("unavailable",it.message)}})
            }
            File(dir,"assets.json").writeText(result.toString())
        }
    }
    fun mark(reason: String) = event("MANUAL_LABEL",buildJsonObject {put("label",reason)})
    fun export(uri: Uri) {
        val dir=current ?: mutable.value.directory?.let(::File) ?: return
        stop()
        io.execute {
            try { context.contentResolver.openOutputStream(uri)?.use { output->ZipOutputStream(output).use {zip->
                dir.listFiles()?.sortedBy {it.name}?.forEach {file->zip.putNextEntry(ZipEntry(file.name))
file.inputStream().use {it.copyTo(zip)}
                zip.closeEntry()}
            }} ?: error("Export konnte nicht geöffnet werden")
            }catch(e:Exception) {mutable.value=mutable.value.copy(error=e.message)}
        }
    }
    override fun close() {stop()
    io.shutdown()}
}
