#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/gpio/consumer.h>
#include <linux/of.h>

#include <linux/proc_fs.h>
#include <linux/slab.h>

#define MAX_USER_SIZE 1024

static struct proc_dir_entry * button_proc = NULL;
struct gpio_desc *gpio;

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
        pr_err("Could not get button state!\n");
        return -EFAULT;
    }

    *off = sizeof(to_copy);

    return sizeof(to_copy);
}

static const struct proc_ops proc_fops = {
    .proc_read = on_read
};

// ======= GPIO ==========

static int mygpio_probe(struct platform_device *pdev)
{    
    pr_info("mygpio: probe()\n");

    gpio = devm_gpiod_get(&pdev->dev, "my", GPIOD_IN);

    if (IS_ERR(gpio)) {
        pr_err("mygpio: konnte GPIO nicht bekommen\n");
        return PTR_ERR(gpio);
    }

    int value = gpiod_get_value(gpio);

    pr_info("mygpio: GPIO value = %d\n", value);

    // create procfs
    button_proc = proc_create("button", 0666, NULL, &proc_fops);
    if (button_proc == NULL) {
            printk("GPIO: Failed to start driver!");
            return -1;
    }

    printk("mygpio: Created procfs in /proc/button!\n");

    return 0;
}

static void mygpio_remove(struct platform_device *pdev) {
    proc_remove(button_proc);
}

static const struct of_device_id mygpio_of_match[] = {
    {
        .compatible = "mycompany,mygpio",
    },
    { }
};

MODULE_DEVICE_TABLE(of, mygpio_of_match);

static struct platform_driver mygpio_driver = {
    .probe = mygpio_probe,
    .remove = mygpio_remove,

    .driver = {
        .name = "mygpio",
        .of_match_table = mygpio_of_match,
    },
};

module_platform_driver(mygpio_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Test");
MODULE_DESCRIPTION("Simple GPIO input driver");
