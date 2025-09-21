# EyeAIVision

![alt text](https://github.com/SimplyExo/eye-ai/blob/eyeaivision/EyeAIVision/Images/Bild.png?raw=true)


EyeAIVision ist die zugehörige Hardware zur EyeAIApp. Das Gerät dient als eine Art mobile Kamera, welche an der Brille der blinden Person befestigt werden kann. Dadurch kann eine Echtzeitaufnahme des Blickfelds des Trägers erfasst werden. Mithilfe eines MJPEG-Videostreams wird das Kamerabild per WiFi ans Smartphone übertragen. Dort wird es von EyeAIApp verarbeitet. 

## Hardware:
Das Gerät wird mithilfe eines ESP32-CAM Mikrocontrollers von Espressif betrieben. Dieses Modell bietet im Gegensatz zu gewöhnlichen ESP32-Boards auch eine Schnittstelle für eine Kamera an. Unser Projekt nutzt das OV2640-Kameramodul mit maximal 2 Megapixeln Auflösung und einem Blickwinkel von 160°. Die Auflösung ist mehr als ausreichend für unser Projekt, da das größte KI-Modell, welches wir nutzen, eine Auflösung von 640x640 Pixeln benötigt. Zudem ermöglicht uns das große Sichtfeld von 160° einen besseren Überblick über die aktuelle Szene im Gegensatz zu gewöhnlichen Kameramodulen.

Betrieben wird der ESP32 mit einem 3,7V Lithium-Ionen-Akku, welcher eine Kapazität von 650 mAh besitzt. Allerdings benötigt der Mikrocontroller eine Spannung von entweder 3,3V oder 5V (diese werden dann auf dem Board per Längsregler AMS1117 auf 3,3V heruntergeregelt). Zur Regelung der Spannung wird ein DC-DC-Wandler verwendet. Da es uns jedoch nicht möglich war, einen solchen Wandler für eine Spannung von 3,3V zu finden, mussten wir auf einen Wandler mit 5V Ausgabespannung ausweichen. Dies hat den Nachteil, dass die Spannung nach dem DC-DC-Wandler durch den Längsregler auf 3,3V reduziert wird, wodurch Energie in Form von Wärme freigesetzt wird. Die Akkulaufzeit verringert sich dadurch.

Der Akku wird per Lademodul (TP4056) für Li-Ion-Akkus aufgeladen. Der Ladevorgang erfolgt per USB Typ-C und kann mit jedem gängigen Ladenetzteil durchgeführt werden. Vorteile hierbei sind vor allem der Nachhaltigkeitsaspekt sowie die einfache Handhabung durch den weit verbreiteten USB-Standard. Außerdem besitzt das Modul eine Schutzschaltung gegen Tiefentladung und Überladung: Im Ernstfall wird die Versorgung zum Verbraucher automatisch unterbrochen. Der einzige Nachteil dieses Moduls liegt darin, dass es von Werk aus nicht möglich ist, den Akku gleichzeitig zu laden und zu nutzen.

Um diese Schwachstelle zu umgehen, wurde eine Bypassschaltung (siehe: https://github.com/DoImant/TP4056-Power-Path-PCB) integriert, die mithilfe eines p-Kanal MOSFETs bestimmt, ob der Mikrocontroller von der Batterie oder vom Ladegerät gespeist wird. Liegt am MOSFET eine Spannung von 5V (ausgehend vom Ladegerät) an, so wird der Stromkreis zwischen Batterie und Controller unterbrochen. Auf diese Weise kann der Akku problemlos geladen werden, während der ESP32 direkt über das USB-Ladegerät mit Energie versorgt wird.

![alt text](https://github.com/SimplyExo/eye-ai/blob/eyeaivision/EyeAIVision/Images/Schaltung.png)
*Komplettes Schaltbild*

Außerdem besitzt die EyeAIVision eine rot-grüne Leuchtdiode, welche den aktuellen Verbindungsstatus zum Smartphone darstellt. Eingelassen ist diese in eine LED-Fassung aus Metall, was uns ermöglichte, sie als Touch-Button zu nutzen. Dies wurde durch das direkte Verbinden des Metalls mit einem der Touch-Pins des ESP32 umgesetzt. Der Button wird zur Steuerung der Android-Applikation verwendet.

Die gesamte Hardware wird von einem mithilfe von 3D-Druck aus PLA gefertigten Gehäuse zusammengehalten und vor Umwelteinflüssen geschützt. Das Ein- bzw. Ausschalten des Geräts erfolgt über einen Schiebeschalter, der an der Unterseite des Gehäuses verklebt ist. Wird dieser umgelegt, so wird die Verbindung zwischen Lademodul und DC-DC-Converter hergestellt oder getrennt. Um Konnektivitätsprobleme zu vermeiden, befindet sich an der Oberseite des Gehäuses eine externe WiFi-Antenne.

![alt text](https://github.com/SimplyExo/eye-ai/blob/eyeaivision/EyeAIVision/Images/Layout.png)
*Bauteillayout*
