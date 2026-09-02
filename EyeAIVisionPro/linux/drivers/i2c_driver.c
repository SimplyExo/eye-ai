#include <linux/module.h>
#include <linux/i2c.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>

#define SET_LED_GREEN            0x01
#define SET_LED_RED              0x02
#define TURN_POWER_OFF           0x03
#define MEASURE_BATTERY_VOLTAGE  0x04

#define DMESG_LABEL "i2c_driver"

#define DEVICE_NAME_LED        "led"
#define DEVICE_NAME_TRANSISTOR "transistor"


// ============================================================
// ATtiny85 slave functionality
// ============================================================

struct i2c_instance {
    struct i2c_client *client;

    // LED
    bool green;

    // Transistor
    bool transistor_on;

    // Battery voltage
    unsigned int voltage_digits;
};


// ============================================================
// LED functionality
// ============================================================

static int set_led_green(struct i2c_instance *data)
{
    int ret;

    ret = i2c_smbus_write_byte(data->client, SET_LED_GREEN);

    if (ret < 0) {
        dev_err(&data->client->dev,
                "Failed to set green LED: %d\n",
                ret);
        return ret;
    }

    data->green = true;

    return 0;
}


static int set_led_red(struct i2c_instance *data)
{
    int ret;

    ret = i2c_smbus_write_byte(data->client, SET_LED_RED);

    if (ret < 0) {
        dev_err(&data->client->dev,
                "Failed to set red LED: %d\n",
                ret);
        return ret;
    }

    data->green = false;

    return 0;
}


// ============================================================
// Transistor functionality
// ============================================================

// WARNING:
// THIS WILL POWER OFF THE SYSTEM BY CUTTING OFF THE POWER SUPPLY!
static int turn_power_off(struct i2c_instance *data)
{
    int ret;

    ret = i2c_smbus_write_byte(
        data->client,
        TURN_POWER_OFF
    );

    if (ret < 0) {
        dev_err(&data->client->dev,
                "Failed to power off system: %d\n",
                ret);
        return ret;
    }

    data->transistor_on = false;

    return 0;
}


// ============================================================
// READ / WRITE
// ============================================================

static ssize_t on_read_led(
    struct file *file,
    char __user *user,
    size_t size,
    loff_t *off)
{
    struct i2c_instance *data;
    char to_copy;

    data = file->private_data;

    if (!data)
        return -ENODEV;

    if (size < 1)
        return -EINVAL;

    if (*off > 0)
        return 0;

    if (data->green)
        to_copy = 'G';
    else
        to_copy = 'R';

    if (copy_to_user(user, &to_copy, 1)) {
        pr_err("%s: Could not get LED state!\n",
               DMESG_LABEL);
        return -EFAULT;
    }

    *off = 1;

    return 1;
}


static ssize_t on_write_led(
    struct file *file,
    const char __user *user,
    size_t size,
    loff_t *off)
{
    struct i2c_instance *data;
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
                "Unknown command (LED): %c\n",
                command);
        return -EINVAL;
    }

    if (ret < 0)
        return ret;

    return size;
}


static ssize_t on_read_transistor(
    struct file *file,
    char __user *user,
    size_t size,
    loff_t *off)
{
    struct i2c_instance *data;
    char to_copy;

    data = file->private_data;

    if (!data)
        return -ENODEV;

    if (size < 1)
        return -EINVAL;

    if (*off > 0)
        return 0;

    if (data->transistor_on)
        to_copy = '1';
    else
        to_copy = '0';

    if (copy_to_user(user, &to_copy, 1)) {
        pr_err("%s: Could not get transistor state!\n",
               DMESG_LABEL);
        return -EFAULT;
    }

    *off = 1;

    return 1;
}


static ssize_t on_write_transistor(
    struct file *file,
    const char __user *user,
    size_t size,
    loff_t *off)
{
    struct i2c_instance *data;
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

    case '0':
        ret = turn_power_off(data);
        break;

    default:
        dev_err(&data->client->dev,
                "Unknown command (transistor): %c\n",
                command);
        return -EINVAL;
    }

    if (ret < 0)
        return ret;

    return size;
}


// ============================================================
// DEVICE NODE
// ============================================================

static char *devnode(
    const struct device *dev,
    umode_t *mode)
{
    if (mode)
        *mode = 0666;

    return NULL;
}


static char *admin_devnode(
    const struct device *dev,
    umode_t *mode)
{
    if (mode)
        *mode = 0600;

    return NULL;
}


// ============================================================
// dev_node
// ============================================================

typedef struct dev_node {
    struct dev_node *next;

    dev_t dev_number;

    struct class *dev_class;
    struct device *dev_device;

    struct cdev cdev;

    /*
     * Wichtig:
     *
     * Dieser Pointer verbindet den cdev mit dem
     * zugehörigen i2c_instance.
     */
    struct i2c_instance *data;

} dev_node;


// ============================================================
// OPEN
// ============================================================

static int on_open(
    struct inode *inode,
    struct file *file)
{
    struct dev_node *node;

    /*
     * inode->i_cdev zeigt auf node->cdev.
     *
     * Deshalb muss container_of() hier
     * struct dev_node verwenden und NICHT
     * struct i2c_instance.
     */
    node = container_of(
        inode->i_cdev,
        struct dev_node,
        cdev
    );

    if (!node)
        return -ENODEV;

    if (!node->data)
        return -ENODEV;

    file->private_data = node->data;

    return 0;
}


// ============================================================
// FILE OPERATIONS
// ============================================================

static const struct file_operations fops_led = {
    .owner = THIS_MODULE,
    .open = on_open,
    .read = on_read_led,
    .write = on_write_led,
};


static const struct file_operations fops_transistor = {
    .owner = THIS_MODULE,
    .open = on_open,
    .read = on_read_transistor,
    .write = on_write_transistor,
};


// ============================================================
// CREATE DEVICE
// ============================================================

static int init_dev_file(
    struct i2c_instance *data,
    const char *dev_name,
    struct class **dev_class,
    struct device **dev_device,
    dev_t *dev_number,
    struct cdev *cdev,
    const struct file_operations *fops,
    bool admin_only)
{
    int ret;

    printk(KERN_INFO
           "%s: Initializing LKM (%s)\n",
           DMESG_LABEL,
           dev_name);


    // --------------------------------------------------------
    // Allocate device number
    // --------------------------------------------------------

    ret = alloc_chrdev_region(
        dev_number,
        0,
        1,
        dev_name
    );

    if (ret < 0) {
        printk(KERN_ERR
               "%s: alloc_chrdev_region failed (%s): %d\n",
               DMESG_LABEL,
               dev_name,
               ret);

        return ret;
    }


    // --------------------------------------------------------
    // Create class
    // --------------------------------------------------------

    *dev_class = class_create(dev_name);

    if (IS_ERR(*dev_class)) {
        ret = PTR_ERR(*dev_class);

        unregister_chrdev_region(
            *dev_number,
            1
        );

        *dev_class = NULL;

        return ret;
    }


    // --------------------------------------------------------
    // Permissions
    // --------------------------------------------------------

    if (admin_only)
        (*dev_class)->devnode = admin_devnode;
    else
        (*dev_class)->devnode = devnode;


    // --------------------------------------------------------
    // Create /dev node
    // --------------------------------------------------------

    *dev_device = device_create(
        *dev_class,
        &data->client->dev,
        *dev_number,
        NULL,
        "%s",
        dev_name
    );

    if (IS_ERR(*dev_device)) {
        ret = PTR_ERR(*dev_device);

        class_destroy(*dev_class);

        unregister_chrdev_region(
            *dev_number,
            1
        );

        *dev_device = NULL;
        *dev_class = NULL;

        return ret;
    }


    // --------------------------------------------------------
    // Initialize cdev
    // --------------------------------------------------------

    cdev_init(cdev, fops);
    cdev->owner = THIS_MODULE;


    // --------------------------------------------------------
    // Add cdev
    // --------------------------------------------------------

    ret = cdev_add(
        cdev,
        *dev_number,
        1
    );

    if (ret < 0) {

        device_destroy(
            *dev_class,
            *dev_number
        );

        class_destroy(*dev_class);

        unregister_chrdev_region(
            *dev_number,
            1
        );

        *dev_device = NULL;
        *dev_class = NULL;

        return ret;
    }


    printk(KERN_INFO
           "%s: Device /dev/%s created successfully\n",
           DMESG_LABEL,
           dev_name);

    return 0;
}


// ============================================================
// DELETE DEVICE
// ============================================================

static void delete_dev_file(
    struct dev_node *node)
{
    if (!node)
        return;

    cdev_del(&node->cdev);

    device_destroy(
        node->dev_class,
        node->dev_number
    );

    class_destroy(node->dev_class);

    unregister_chrdev_region(
        node->dev_number,
        1
    );
}


// ============================================================
// LINKED LIST
// ============================================================

static dev_node *dev_head = NULL;


static dev_node *add_device(
    dev_node *head,
    const char *device_name,
    struct i2c_instance *instance,
    const struct file_operations *fops,
    bool admin_only)
{
    dev_node *node;
    int ret;

    node = kzalloc(
        sizeof(*node),
        GFP_KERNEL
    );

    if (!node) {
        printk(KERN_ERR
               "%s: Couldn't allocate dev_node\n",
               DMESG_LABEL);

        return NULL;
    }

    /*
     * Speichere die Verbindung:
     *
     * node -> i2c_instance
     */
    node->data = instance;


    ret = init_dev_file(
        instance,
        device_name,
        &node->dev_class,
        &node->dev_device,
        &node->dev_number,
        &node->cdev,
        fops,
        admin_only
    );

    if (ret < 0) {
        kfree(node);
        return NULL;
    }


    // New node goes to the front.
    node->next = head;

    return node;
}


static void clear_list(dev_node *head)
{
    while (head != NULL) {

        dev_node *next = head->next;

        delete_dev_file(head);

        kfree(head);

        head = next;
    }
}


// ============================================================
// PROBE
// ============================================================

static int led_driver_probe(
    struct i2c_client *client)
{
    struct i2c_instance *data;
    dev_node *node;
    int ret;


    // --------------------------------------------------------
    // Allocate driver data
    // --------------------------------------------------------

    data = devm_kzalloc(
        &client->dev,
        sizeof(*data),
        GFP_KERNEL
    );

    if (!data)
        return -ENOMEM;


    data->client = client;
    data->green = false;
    data->transistor_on = true;


    i2c_set_clientdata(
        client,
        data
    );


    dev_info(
        &client->dev,
        "ATtiny85 detected: addr=0x%02x\n",
        client->addr
    );


    // --------------------------------------------------------
    // Set initial LED state
    // --------------------------------------------------------

    ret = set_led_red(data);

    if (ret < 0)
        return ret;


    // --------------------------------------------------------
    // Create LED device
    // --------------------------------------------------------

    node = add_device(
        dev_head,
        DEVICE_NAME_LED,
        data,
        &fops_led,
        false
    );

    if (!node) {
        dev_err(
            &client->dev,
            "Failed to create LED device\n"
        );

        return -ENOMEM;
    }

    dev_head = node;


    // --------------------------------------------------------
    // Create transistor device
    // --------------------------------------------------------

    node = add_device(
        dev_head,
        DEVICE_NAME_TRANSISTOR,
        data,
        &fops_transistor,
        true
    );

    if (!node) {

        dev_err(
            &client->dev,
            "Failed to create transistor device\n"
        );

        clear_list(dev_head);
        dev_head = NULL;

        return -ENOMEM;
    }

    dev_head = node;


    dev_info(
        &client->dev,
        "Character devices initialized\n"
    );

    return 0;
}


// ============================================================
// REMOVE
// ============================================================

static void led_driver_remove(
    struct i2c_client *client)
{
    struct i2c_instance *data;

    data = i2c_get_clientdata(client);

    if (!data)
        return;


    clear_list(dev_head);

    dev_head = NULL;


    dev_info(
        &client->dev,
        "I2C device removed\n"
    );
}


// ============================================================
// I2C DEVICE ID
// ============================================================

static const struct i2c_device_id led_driver_ids[] = {
    { "eyeai_i2c", 0 },
    { }
};

MODULE_DEVICE_TABLE(
    i2c,
    led_driver_ids
);


// ============================================================
// DEVICE TREE
// ============================================================

static const struct of_device_id led_driver_of_match[] = {
    {
        .compatible = "eyeaivision,eyeai_i2c",
    },
    { }
};

MODULE_DEVICE_TABLE(
    of,
    led_driver_of_match
);


// ============================================================
// I2C DRIVER
// ============================================================

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
MODULE_DESCRIPTION(
    "Driver for controlling i2c components on the HAT"
);
