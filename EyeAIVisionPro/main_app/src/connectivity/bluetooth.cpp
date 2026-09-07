#include <connectivity/bluetooth.hpp>
#include <QBluetoothHostInfo>
#include <QBluetoothLocalDevice>

QString bluetooth::get_bt_mac() {
    QList<QBluetoothHostInfo> localAdapters = QBluetoothLocalDevice::allDevices();
    
    if (localAdapters.size() > 0)
        return localAdapters[0].address().toString();   // rpi only has one adapter
    else
        return "00:00:00:00:00:00";
}
