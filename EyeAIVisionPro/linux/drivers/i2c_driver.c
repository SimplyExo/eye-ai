#include <linux/module.h>
#include <linux/i2c.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>

#define SET_LED_GREEN 0x01
#define SET_LED_RED 0x02
#define TURN_POWER_OFF 0x03
#define MEASURE_BATTERY_VOLTAGE 0x04

#define DMESG_LABEL "i2c_driver"
#define DEVICE_NAME "led"
#define CLASS_NAME "led_class"

// ====== ATtiny85 slave Functionality =======

struct led_instance {
    struct i2c_client *client;
    bool green;
    struct cdev cdev;
};


static int set_led_green(struct led_instance *data) {
    int ret = i2c_smbus_write_byte(data->client, SET_LED_GREEN);
    if (ret < 0) {
        dev_err(&data->client->dev,
                "Failed to set green LED: %d\n", ret);
    }
    
    data->green = true;
    return ret;
}

static int set_led_red(struct led_instance *data) {
    int ret = i2c_smbus_write_byte(data->client, SET_LED_RED);
    if (ret < 0) {
        dev_err(&data->client->dev,
                "Failed to set red LED: %d\n", ret);
    }
    
    data->green = false;
    return ret;
}

// WARNING: THIS WILL POWER OFF THE SYSTEM BY CUTTING OFF THE POWER SUPPLY!
static int turn_power_off(struct i2c_client *client) {
    int ret = i2c_smbus_write_byte(client, TURN_POWER_OFF);
    if (ret < 0) {
        dev_err(&client->dev,
                "Failed to power off system: %d\n", ret);
    }
    
    return ret;
}

// ====== READ / WRITE EVENTS =======

static ssize_t on_read(struct file *file,
                char __user *user,
                size_t size,
                loff_t *off)
{
    struct led_instance *data;
    data = file->private_data;

    char to_copy;

    if (data->green)
        to_copy = 'G';
    else 
        to_copy = 'R';

    if (!data)
        return -ENODEV;

    if (*off > 0)
        return 0;

    if (copy_to_user(user, &to_copy, sizeof(to_copy))) {
        pr_err("%s: Could not get led state!\n", DMESG_LABEL);
        return -EFAULT;
    }

    *off = sizeof(to_copy);

    return sizeof(to_copy);
}

static ssize_t on_write(struct file *file,
                        const char __user *user,
                        size_t size,
                        loff_t *off)
{
    struct led_instance *data;
    char command;
    int ret;

    data = file->private_data;

    if (!data)
        return -ENODEV;

    if (size < 1)
        return -EINVAL;

    if (copy_from_user(&command, user, 1))
        return -EFAULT;

    switch (command) {

    case 'G':
    case 'g':
        ret = set_led_green(data);
        break;

    case 'R':
    case 'r':
        ret = set_led_red(data);
        break;

    default:
        dev_err(&data->client->dev,
                "Unknown command: %c\n",
                command);
        return -EINVAL;
    }

    if (ret < 0)
        return ret;

    return size;
}

static int on_open(struct inode *inode, struct file *file)
{
    struct led_instance *data;

    data = container_of(inode->i_cdev,
                        struct led_instance,
                        cdev);

    file->private_data = data;

    return 0;
}

// ====== DEVICE FILE =======

static dev_t dev_number;
static struct class *led_class = NULL;
static struct device *led_device = NULL;

static char *led_devnode(const struct device *dev, umode_t *mode)
{
    if (mode)
        *mode = 0666;

    return NULL;
}

static const struct file_operations fops = {
    .owner = THIS_MODULE,
    .read = on_read,
    .open = on_open,
    .write = on_write
};

static int init_dev_file(struct led_instance *data)
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

    led_class = class_create(CLASS_NAME);
    if (IS_ERR(led_class)) {
        unregister_chrdev_region(dev_number, 1);
        printk(KERN_ALERT "%s: Failed to register device class\n", DMESG_LABEL);
        return PTR_ERR(led_class);
    }

    // setting file permissions
    led_class->devnode = led_devnode;

    printk(KERN_INFO "%s: device class registered correctly\n", DMESG_LABEL);

    led_device = device_create(led_class,
                                   &data->client->dev,
                                   dev_number,
                                   NULL,
                                   DEVICE_NAME);

    if (IS_ERR(led_device)) {
        ret = PTR_ERR(led_device);

        class_destroy(led_class);
        unregister_chrdev_region(dev_number, 1);

        printk(KERN_ALERT "%s: Failed to create the device\n", DMESG_LABEL);
        return ret;
    }

    printk(KERN_INFO "%s: device created correctly\n", DMESG_LABEL);

    cdev_init(&data->cdev, &fops);
    data->cdev.owner = THIS_MODULE;

    ret = cdev_add(&data->cdev, dev_number, 1);
    if (ret < 0) {
        device_destroy(led_class, dev_number);
        class_destroy(led_class);
        unregister_chrdev_region(dev_number, 1);

        printk(KERN_ALERT "%s: Failed to add cdev: %d\n", DMESG_LABEL, ret);
        return ret;
    }

    return 0;
}


static void delete_dev_file(struct led_instance *data)
{
    cdev_del(&data->cdev);
    device_destroy(led_class, dev_number);
    class_destroy(led_class);
    unregister_chrdev_region(dev_number, 1);

    pr_info("%s: Disabled LKM\n", DMESG_LABEL);
}

// ====== START / STOP DRIVER =======

static int led_driver_probe(struct i2c_client *client)
{
    struct led_instance *data;
    int ret;

    data = devm_kzalloc(&client->dev,
                        sizeof(*data),
                        GFP_KERNEL);

    if (!data)
        return -ENOMEM;

    data->client = client;
    data->green = false;

    i2c_set_clientdata(client, data);

    dev_info(&client->dev,
             "ATtiny85 detected: addr=0x%02x\n",
             client->addr);

    // set led red
    set_led_red(data);

    ret = init_dev_file(data);

    if (ret < 0)
        return ret;

    return 0;
}

static void led_driver_remove(struct i2c_client *client)
{
    struct led_instance *data;

    data = i2c_get_clientdata(client);

    if (!data)
        return;

    delete_dev_file(data);

    dev_info(&client->dev,
             "I2C device removed\n");
}

static const struct i2c_device_id led_driver_ids[] = {
    { "eyeai_i2c", 0 },
    { }
};

MODULE_DEVICE_TABLE(i2c, led_driver_ids);

static const struct of_device_id led_driver_of_match[] = {
    {
        .compatible = "eyeaivision,eyeai_i2c",
    },
    { }
};

MODULE_DEVICE_TABLE(of, led_driver_of_match);

static struct i2c_driver led_driver_driver = {
    .driver = {
        .name = "eyeai_i2c",
        .of_match_table = led_driver_of_match,
    },
    .probe = led_driver_probe,
    .remove = led_driver_remove,
    .id_table = led_driver_ids,
};

module_i2c_driver(led_driver_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Thomas Fritzler");
MODULE_DESCRIPTION("Driver for controlling i2c components on the HAT");