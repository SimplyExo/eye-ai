package com.algorithmic_alliance.eyeaiapp.tts

// Native TTS can only stop the whole queue. Never cancel another owner's queued utterance.
class SpeechOwnership {
    private val owners=linkedMapOf<String,String>()
    private var active: String?=null
    @Synchronized fun enqueue(id: String,owner: String,flush: Boolean) {if(flush) {owners.clear()
active=null}
if(owners.isEmpty())active=id
    owners[id]=owner}
    @Synchronized fun started(id: String) {if(id in owners)active=id}
    @Synchronized fun finished(id: String) {owners.remove(id)
    if(active==id)active=null}
    @Synchronized fun canCancel(prefix: String)=owners[active]?.startsWith(prefix)==true&&owners.values.all {it.startsWith(prefix)}
    @Synchronized fun clear() {owners.clear()
    active=null}
}
