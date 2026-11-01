obj-m += eyeai-button.o
eyeai-button-objs := button_driver.o

modules:
	$(MAKE) -C $(KERNEL_SRC) M=$(shell pwd) modules

modules_install:
	$(MAKE) -C $(KERNEL_SRC) M=$(shell pwd) modules_install

clean:
	$(MAKE) -C $(KERNEL_SRC) M=$(shell pwd) clean
