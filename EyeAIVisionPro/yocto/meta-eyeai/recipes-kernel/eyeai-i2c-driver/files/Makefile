obj-m += eyeai-i2c.o
eyeai-i2c-objs := i2c_driver.o

modules:
	$(MAKE) -C $(KERNEL_SRC) M=$(shell pwd) modules

modules_install:
	$(MAKE) -C $(KERNEL_SRC) M=$(shell pwd) modules_install

clean:
	$(MAKE) -C $(KERNEL_SRC) M=$(shell pwd) clean
