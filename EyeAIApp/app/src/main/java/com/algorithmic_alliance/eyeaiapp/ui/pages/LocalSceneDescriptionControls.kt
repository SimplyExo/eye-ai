package com.algorithmic_alliance.eyeaiapp.ui.pages

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.algorithmic_alliance.eyeaiapp.scenenarrator.LocalSceneDescriptions
import com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntime
import com.algorithmic_alliance.eyeaiapp.ui.MainViewModel
import kotlinx.coroutines.flow.StateFlow

@Composable
fun LocalSceneDescriptionControls(runtime: EyeAIRuntime) {
    SceneDescriptionControls(runtime.localSceneDescriptions.state,runtime::setLocalSceneDescriptionsEnabled,
        runtime::speakLocalSceneDescription, runtime.settings.isR4Mode)
}

// Detailed diagnostics stay in a modal.
@Composable
fun R4Diagnostics(runtime: EyeAIRuntime) {
    val state by runtime.localSceneDescriptions.state.collectAsStateWithLifecycle()
    var open by remember { mutableStateOf(false) }
    if(!state.enabled) return
    val context=LocalContext.current.applicationContext
    val names=remember(context) { runCatching { context.assets.open("coco_german.names").bufferedReader().use { it.readLines() } }.getOrDefault(emptyList()) }
    TextButton(onClick={open=true},modifier=Modifier.padding(horizontal=12.dp)) { Text("R4-Diagnose öffnen") }
    if(open) AlertDialog(
        onDismissRequest={open=false},
        title={Text("R4-Beziehungen")},
        confirmButton={TextButton(onClick={open=false}) {Text("Schließen")}},
        text={Column(Modifier.heightIn(max=440.dp).verticalScroll(rememberScrollState())) {
            Text(state.model,style=MaterialTheme.typography.bodySmall)
            Text("Experimentelle Modellvermutungen. Scores sind kein Nachweis physischer Wahrheit.",style=MaterialTheme.typography.bodySmall)
            if(state.r4Diagnostics.isEmpty()) Text("Noch keine Kandidaten vorhanden.")
            state.r4Diagnostics.firstOrNull()?.let {Text("Snapshot ${it.snapshotSha256.take(12)} · Aufnahme ${state.sourceFrameId}",style=MaterialTheme.typography.bodySmall)}
            for(candidate in state.r4Diagnostics) {
                HorizontalDivider(Modifier.padding(vertical=8.dp))
                val kind=if(candidate.calibrated) "kalibriert" else "roh, unkalibriert"
                val score=String.format(java.util.Locale.ROOT,"%.3f",candidate.score)
                val subject=names.getOrNull(candidate.subjectClass-1) ?: "Klasse ${candidate.subjectClass}"
                val obj=names.getOrNull(candidate.objectClass-1) ?: "Klasse ${candidate.objectClass}"
                Text("$subject #${candidate.subjectId} → $obj #${candidate.objectId}")
                Text("${candidate.relation} · $score ($kind)",style=MaterialTheme.typography.bodySmall)
                Text(candidate.reason,style=MaterialTheme.typography.bodySmall)
            }
        }}
    )
}

@Composable
fun LocalSceneDescriptionControls(viewModel: MainViewModel) {
    SceneDescriptionControls(viewModel.localSceneDescriptionState,viewModel::setLocalSceneDescriptionsEnabled,
        viewModel::speakLocalSceneDescription, viewModel.isR4Mode(), showSettingsHint = false)
}

@Composable
private fun SceneDescriptionControls(
    source: StateFlow<LocalSceneDescriptions.State>,
    setEnabled: (Boolean) -> Unit,
    speak: () -> Unit,
    isR4Mode: Boolean,
    showSettingsHint: Boolean = true,
) {
    val state by source.collectAsStateWithLifecycle()
    if (!isR4Mode) {
        if (showSettingsHint) Text("Szenenbeschreibung in den Einstellungen auswählen.",
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
            style = MaterialTheme.typography.bodySmall)
        return
    }
    var open by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth().padding(horizontal=12.dp,vertical=4.dp)) {
        Row(modifier=Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.SpaceBetween) {
            Text("Automatische Szenenorientierung",Modifier.weight(1f))
            Switch(checked=state.enabled,onCheckedChange=setEnabled)
        }
        Row {
            TextButton(onClick=speak,enabled=state.text.isNotBlank()) {Text("Erneut vorlesen")}
            TextButton(onClick={open=true}) {Text("Beschreibung öffnen")}
        }
        Text(state.phase,maxLines=1,overflow=TextOverflow.Ellipsis,style=MaterialTheme.typography.bodySmall)
    }
    if(open) AlertDialog(
        onDismissRequest={open=false},
        title={Text("Umgebungsbeschreibung")},
        confirmButton={TextButton(onClick={open=false}) {Text("Schließen")}},
        dismissButton={TextButton(onClick=speak,enabled=state.text.isNotBlank()) {Text("Vorlesen")}},
        text={Column(Modifier.heightIn(max=440.dp).verticalScroll(rememberScrollState()),verticalArrangement=Arrangement.spacedBy(12.dp)) {
            SelectionContainer {Text(state.text.ifBlank {"Noch keine Beschreibung vorhanden."})}
            HorizontalDivider()
            Text(state.phase,style=MaterialTheme.typography.bodySmall)
            Text(state.model,style=MaterialTheme.typography.bodySmall)
            state.latencyMs?.let {Text("${it} ms · Aufnahme ${state.sourceFrameId}",style=MaterialTheme.typography.bodySmall)}
            Text("Eine Beschreibung nach deutlicher Szenenänderung, danach Ruhe. Kamera in Blickrichtung halten. Verarbeitung auf diesem Gerät.",style=MaterialTheme.typography.bodySmall)
            state.error?.let {Text(it,style=MaterialTheme.typography.bodySmall,color=MaterialTheme.colorScheme.error)}
        }}
    )
}

// Recording is explicit and local, a modal keeps the image previews large.
@Composable
fun ContextDiagnostics(runtime: EyeAIRuntime) {
    var open by remember { mutableStateOf(false) }
    val controller=runtime.localSceneDescriptions
    val contextState by controller.state.collectAsStateWithLifecycle()
    val recording by controller.recorder.state.collectAsStateWithLifecycle()
    val export=androidx.activity.compose.rememberLauncherForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.CreateDocument("application/zip")) {uri->
        if(uri!=null)controller.recorder.export(uri)
    }
    TextButton(onClick={open=true}) {Text(if(recording.active)"Kontextdiagnose · Aufnahme" else "Kontextdiagnose")}
    if(open)AlertDialog(onDismissRequest={open=false},title={Text("Kontext und Auslöser")},
        confirmButton={TextButton(onClick={open=false}) {Text("Schließen")}},
        text={Column(Modifier.heightIn(max=460.dp).verticalScroll(rememberScrollState()),verticalArrangement=Arrangement.spacedBy(8.dp)) {
            Text("Episode ${contextState.episodeId} · ${contextState.contextReason}")
            Text("Startschwellen sind ein unkalibrierter Prototyp. Andere Sprachdialoge werden nicht unterbrochen.",style=MaterialTheme.typography.bodySmall)
            val scores=contextState.contextScores
            fun number(value: Double?)=value?.let {String.format(java.util.Locale.ROOT,"%.2f",it)} ?: "–"
            Text("Änderung ${number(scores.combined)} · Abdeckung ${number(scores.coverage)} · ${scores.support} stützende Gruppen")
            val values=mapOf("objects" to scores.objects,"visual" to scores.visual,"depth" to scores.depth,"semantics" to scores.semantics)
            val labels=mapOf("objects" to "Objekte","visual" to "Bild","depth" to "Tiefe","semantics" to "Flächen")
            for((key,value) in values) Text("${labels[key]}: Änderung ${number(value)} · Qualität ${number(scores.reliability[key])} · Gewicht ${number(scores.weights[key])}",style=MaterialTheme.typography.bodySmall)
            Text("Ergänzungen ${scores.additions} · Verluste ${scores.removals} · Kandidaten ${contextState.objectHistory.pending} · kurzlebig ${contextState.objectHistory.transient}",style=MaterialTheme.typography.bodySmall)
            if(contextState.collecting)Text("Sammelfenster: noch höchstens ${contextState.collectionRemainingMs} ms")
            Text("Aufnahmekanäle: ${contextState.captureChannels}",style=MaterialTheme.typography.bodySmall)
            Row(verticalAlignment=androidx.compose.ui.Alignment.CenterVertically) {
                Text("Schattenanalyse · ohne automatische Ansage",Modifier.weight(1f))
                Switch(contextState.shadow,controller::setShadow)
            }
            HorizontalDivider()
            Text("Diagnose speichert Kamerabilder und Backend-Rohdaten ausschließlich auf diesem Gerät. Maximal 10 Minuten / 512 MiB. Export erfolgt nur über die Dateiauswahl.",style=MaterialTheme.typography.bodySmall)
            Text("${recording.bytes/1024/1024} MiB · ${recording.dropped} verworfene Datensätze")
            Button(onClick={runtime.setContextRecording(!recording.active)}) {Text(if(recording.active)"Aufnahme stoppen" else "Lokale Aufnahme starten")}
            TextButton(onClick={export.launch("eyeai-context-${System.currentTimeMillis()}.zip")},enabled=recording.directory!=null) {Text("Aufnahme exportieren")}
            TextButton(onClick={controller.markContext("DESCRIPTION_NEEDED")},enabled=recording.active) {Text("Hier wäre eine Beschreibung sinnvoll")}
            TextButton(onClick={controller.markContext("UNNECESSARY_TRIGGER")},enabled=recording.active) {Text("Letzte Ansage war unnötig")}
            recording.error?.let {Text(it,color=MaterialTheme.colorScheme.error)}
        }})
}
