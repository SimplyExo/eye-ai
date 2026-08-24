#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/gpio/consumer.h>
#include <linux/of.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>

#define DMESG_LABEL "hat_button_driver"

#define DEVICE_NAME "button"
#define CLASS_NAME "button_class"

struct gpio_desc *gpio;

// ========= Prototypes ==========
ssize_t on_read(struct file *file,
                char __user *user,
                size_t size,
                loff_t *off);

static int init_dev_file(void);


// ========= Events =========
ssize_t on_read(struct file *file,
                char __user *user,
                size_t size,
                loff_t *off)
{
    char to_copy = '0' + !gpiod_get_value(gpio);    // gpiod_get_value is 0 if HIGH and 1 if LOW

    if (*off > 0)
        return 0;

    if (copy_to_user(user, &to_copy, sizeof(to_copy))) {
        pr_err("%s: Could not get button state!\n", DMESG_LABEL);
        return -EFAULT;
    }

    *off = sizeof(to_copy);

    return sizeof(to_copy);
}

// ======= GPIO ========

static int init_gpio(struct platform_device *pdev) {
    gpio = devm_gpiod_get(&pdev->dev, "button", GPIOD_IN);

    if (IS_ERR(gpio)) {
        pr_err("%s: couldn't get GPIO\n", DMESG_LABEL);
        return PTR_ERR(gpio);
    }

    return 0;
}

// ======= device file ========

static dev_t dev_number;
static struct class *button_class = NULL;
static struct device *button_device = NULL;
static struct cdev button_cdev;

static char *button_devnode(const struct device *dev, umode_t *mode)
{
    if (mode)
        *mode = 0666;

    return NULL;
}

static const struct file_operations fops = {
    .owner = THIS_MODULE,
    .read = on_read,
};

static int init_dev_file(void)
{
    int ret;

    printk(KERN_INFO "%s: Initializing LKM\n", DMESG_LABEL);

    ret = alloc_chrdev_region(&dev_number, 0, 1, DEVICE_NAME);
    if (ret < 0) {
        printk(KERN_ALERT "%s: failed to register a major number\n", DMESG_LABEL);
        return ret;
    }

    printk(KERN_INFO "%s: registered correctly with major number %d\n",
           DMESG_LABEL,
           MAJOR(dev_number));

    button_class = class_create(CLASS_NAME);
    if (IS_ERR(button_class)) {
        unregister_chrdev_region(dev_number, 1);
        printk(KERN_ALERT "%s: Failed to register device class\n", DMESG_LABEL);
        return PTR_ERR(button_class);
    }

    // setting file permissions
    button_class->devnode = button_devnode;

    printk(KERN_INFO "%s: device class registered correctly\n", DMESG_LABEL);

    button_device = device_create(button_class,
                                   NULL,
                                   dev_number,
                                   NULL,
                                   DEVICE_NAME);

    if (IS_ERR(button_device)) {
        ret = PTR_ERR(button_device);

        class_destroy(button_class);
        unregister_chrdev_region(dev_number, 1);

        printk(KERN_ALERT "%s: Failed to create the device\n", DMESG_LABEL);
        return ret;
    }

    printk(KERN_INFO "%s: device created correctly\n", DMESG_LABEL);

    cdev_init(&button_cdev, &fops);

    ret = cdev_add(&button_cdev, dev_number, 1);
    if (ret < 0) {
        device_destroy(button_class, dev_number);
        class_destroy(button_class);
        unregister_chrdev_region(dev_number, 1);

        printk(KERN_ALERT "%s: Failed to add cdev: %d\n", DMESG_LABEL, ret);
        return ret;
    }

    return 0;
}


static void delete_dev_file(void)
{
    cdev_del(&button_cdev);
    device_destroy(button_class, dev_number);
    class_destroy(button_class);
    unregister_chrdev_region(dev_number, 1);

    pr_info("%s: Disabled LKM\n", DMESG_LABEL);
}


// ======= START / STOP ==========

static int hat_button_driver_probe(struct platform_device *pdev)
{
    // setup gpio
    pr_info("%s: probe()\n", DMESG_LABEL);

    int ret;
    ret = init_gpio(pdev);

    if (ret) {
        return ret;
    }

    // setup dev-file
    ret = init_dev_file();
    if (ret) {
        return ret;
    }

    printk("%s: Created device file in /dev/button!\n", DMESG_LABEL);

    return 0;

}

static void hat_button_driver_remove(struct platform_device *pdev) {
    delete_dev_file();
}

// ========== Register Driver ============

static const struct of_device_id hat_button_driver_of_match[] = {
    {
        .compatible = "eyeaivision,button",
    },
    { }
};

MODULE_DEVICE_TABLE(of, hat_button_driver_of_match);

static struct platform_driver hat_button_driver_driver = {
    .probe = hat_button_driver_probe,
    .remove = hat_button_driver_remove,

    .driver = {
        .name = "button_driver",
        .of_match_table = hat_button_driver_of_match,
    },
};

module_platform_driver(hat_button_driver_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Thomas Fritzler");
MODULE_DESCRIPTION("Driver for accessing on/off button state via /dev/button");
