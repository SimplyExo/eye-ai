#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/gpio/consumer.h>
#include <linux/of.h>

#include <linux/proc_fs.h>
#include <linux/slab.h>

static struct proc_dir_entry * button_proc = NULL;
struct gpio_desc *gpio;

// ========= Prototypes ==========
ssize_t on_read(struct file *file,
                char __user *user,
                size_t size,
                loff_t *off);




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
        pr_err("hat_button_driver: Could not get button state!\n");
        return -EFAULT;
    }

    *off = sizeof(to_copy);

    return sizeof(to_copy);
}

static const struct proc_ops proc_fops = {
    .proc_read = on_read
};

// ======= GPIO ==========

static int hat_button_driver_probe(struct platform_device *pdev)
{
    // setup gpio
    pr_info("hat_button_driver: probe()\n");

    gpio = devm_gpiod_get(&pdev->dev, "button", GPIOD_IN);

    if (IS_ERR(gpio)) {
        pr_err("hat_button_driver: couldn't get GPIO\n");
        return PTR_ERR(gpio);
    }

    // create procfs
    button_proc = proc_create("button", 0666, NULL, &proc_fops);
    if (button_proc == NULL) {
            printk("hat_button_driver: Failed to start driver!");
            return -EFAULT;
    }

    printk("hat_button_driver: Created procfs in /proc/button!\n");

    return 0;
}

static void hat_button_driver_remove(struct platform_device *pdev) {
    proc_remove(button_proc);
}

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
MODULE_DESCRIPTION("Driver for accessing on/off button state via /proc/button");
