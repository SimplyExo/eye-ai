#pragma once

#include <qobject.h>

class bluetooth : QObject {
    Q_OBJECT

    public:
        static QString get_bt_mac();
};