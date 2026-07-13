package com.algorithmic_alliance.eyeaiapp.data

import com.algorithmic_alliance.eyeaiapp.R

object UIDataSource {

    val NEEDED_PERMISSIONS = listOf<Map<String, Any>>(
        /* EXPLANATION HOW TO ADD NEW PERMISSION
        mapOf(
            "permissionName": Name of the permission (String)
            "permissionExplanation": Explain to the user why the app needs that permission (String)
            "icon": Icon matching that permission (Int)
            "iconDescription": Describing the Icon for the semantics (String)
            "permissionDeclineSemantic": semantic for the decline button (String),
            "permissionAcceptSemantic": semantic for the accept button (String),
            "confirmPermissionDeclineExplanation": Explain the effects of declining that permission for the conformation dialog (String)
            "confirmPermissionDeclineSemantic": semantic for confirm declining button (String)
        ),
         */
        mapOf(
            "permissionName" to "Kamera",
            "permissionExplanation" to """Damit die KI die Umgebung analysieren kann, ist es notwendig, dass die App auf die System-Kamera zugreifen kann. 
                |Die Kamerabilder werden genutzt, um Entfernungen zu Objekten zu bestimmen und um vorhandene Objekte im Raum zu erkennen. 
                |Diese Informationen werden dann per Audio ausgegeben.""".trimMargin(),
            "icon" to R.drawable.photo_camera_24px,
            "iconDescription" to "Kamera Icon",
            "permissionDeclineSemantic" to "Zugriff auf Kamera ablehnen.",
            "permissionAcceptSemantic" to "Zugriff auf Kamera gestatten.",
            "confirmPermissionDeclineExplanation" to "Wenn Sie den Zugriff auf die Kamera ablehnen, können sie die App nicht benutzen.",
            "confirmPermissionDeclineSemantic" to "Zugriff auf Kamera trotzdem ablehnen. Die App wird geschlossen."
        ),
        mapOf(
            "permissionName" to "Mikrophon",
            "permissionExplanation" to "Um per Sprachbefehl mit der App zu interagieren, ist es notwendig, zugriff auf das System-Mikrophon zu erteilen.",
            "icon" to R.drawable.mic_24px,
            "iconDescription" to "Mikrophon Icon",
            "permissionDeclineSemantic" to "Zugriff auf Mikrophon ablehnen.",
            "permissionAcceptSemantic" to "Zugriff auf Mikrophon gestatten.",
            "confirmPermissionDeclineExplanation" to "Wenn Sie den Zugriff auf das Mikrophon ablehnen, können sie die App nicht benutzen.",
            "confirmPermissionDeclineSemantic" to "Zugriff auf Mikrophon trotzdem ablehnen. Die App wird geschlossen."
        )
    )

    const val INFORMATION_NOT_FOUND = "Die Information konnte nicht geladen werden. Wir bitten um Entschuldigung."

    val ICON_NOT_FOUND = R.drawable.error_24px

    const val RETURN_SEMANTIC = "Zurück."

}