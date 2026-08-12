obj-m += button_driver.o

KDIR := /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)
OUT  := $(PWD)/build

all:
	make -C $(KDIR) M=$(PWD) MO=$(OUT) modules

clean:
	make -C $(KDIR) M=$(PWD) MO=$(OUT) clean