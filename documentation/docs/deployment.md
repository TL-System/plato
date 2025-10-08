# Deployment Guide

## General notes on deploying the Plato federated learning server in the cloud

The Plato federated learning server is designed to use Socket.IO over HTTP and HTTPS, and can be easily deployed in a production server environment in the public cloud.

To deploy such a production federated learning server in a virtual machine from any public cloud provider, a `nginx` web server will first need to be installed to serve as a reverse proxy server. To install the `nginx` web server in Ubuntu 24.04, follow Step 1 in the guide on [How To Install Nginx on Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-22-04).

Once `nginx` has been installed and tested, use the following configuration file in `/etc/nginx/sites-available/example.com` (where `example.com` is the domain name for the server):

```
server {
    listen      80;
    listen      443 ssl;
    server_name example.com www.example.com;
    root /home/username/example.com;
    index index.html index.htm index.php;

    ssl_certificate /etc/nginx/ssl/example.cer;
    ssl_certificate_key /etc/nginx/ssl/example.key;

    location / {
        try_files $uri $uri/ =404;
    }

    location /socket.io {
        proxy_pass http://127.0.0.1:8000/socket.io;
        proxy_redirect off;
        proxy_buffering off;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
    }

    location ~ /\.ht {
        deny all;
    }
}
```

This configuration file assumes that the certificate and private key for establishing a HTTPS connection to the server are stored as `/etc/nginx/ssl/example.cer` and `/etc/nginx/ssl/example.key`, respectively. It also assumes that all static content for the website is stored at `/home/username/example.com`.

If there is a need for using load balancing available from `nginx`, sticky sessions must be used for Socket.IO:

!!! info "Load Balancing Configuration"
    If there is a need for using load balancing available from `nginx`, sticky sessions must be used for Socket.IO:

    ```nginx
    upstream example_servers {
        ip_hash; # enabling sticky sessions for Socket.IO
        server 127.0.0.1:8000;
        server 127.0.0.1:8001;
    }

    server {
        ...
        location /socket.io {
            proxy_pass http://example_servers;
            ...
        }
    }
    ```

After the configuration file is created, create a symbolic link in `/etc/nginx/sites-enabled`:

```bash
sudo ln -s /etc/nginx/sites-available/example.com /etc/nginx/sites-enabled/
```

Then test and restart the web server:

```bash
sudo nginx -t
sudo systemctl restart nginx
```

The Plato federated learning server can be started as usual. For example:

```bash
cd examples/customized
uv run custom_server.py
```

On the clients, make sure that the configuration file contains the correct domain name of the production server deployed in the cloud:

```
server:
    address: example.com
    use_https: true
```

And then run the clients as usual. For example:
```bash
uv run custom_client.py -i 1
```

There is no need to specify the port number for production servers deployed in the cloud.

## Deploying a Plato Federated Learning Server with DigitalOcean

Here is some more detailed documentation on deploying a Plato federated learning server in one of the production environments: DigitalOcean.

### Prerequisites

#### Creating your droplet

First thing first, create an account on [DigitalOcean](https://www.digitalocean.com) if you haven't, and sign in.

For your future convenience, follow [this tutorial](https://docs.digitalocean.com/products/droplets/how-to/add-ssh-keys/) to use SSH keys with your **Droplets** (DigitalOcean Droplets are Linux-based virtual machines that run on top of virtualized hardware. Each Droplet you create is a new server you can use.)

Then go back to your DigitalOcean homepage (control panel), click the green button `Create` on the upper right corner and choose `Droplets`.  Here we need to create a droplet to use it as your Plato federated learning server. Choose Ubuntu 24.04 (LTS) x64 image.

**Note**: Don't forget to check `IPv6` in the `Select additional options` to enable public IPv6 networking, so that you can SSH to your server right after you create it.

!!! warning "Important"
    Don't forget to check `IPv6` in the `Select additional options` to enable public IPv6 networking, so that you can SSH to your server right after you create it.

After creating your droplet, click `Droplets` under `MANAGE` on the left-hand side of your control panel, you will see the name of the droplet you just create. Click it and you will see all the information of it, including its IP address.

Open terminal on your local machine and you should be able to log in to your droplet as the `root` user:

```
$ ssh root@<IP address of your droplet>
```

Then configure a regular user account by following [this guide](https://www.digitalocean.com/community/tutorials/initial-server-setup-with-ubuntu-20-04).

#### Purchasing a domain name

Purchase a domain name that will be used later. A recommended place to purchase it is [Namecheap](https://namecheap.com), where you can easily search and buy affordable domain names.

### Installing Nginx

To deploy a production federated learning server in a virtual machine from any public cloud provider, a `nginx` web server will first need to be installed to serve as a reverse proxy server.

To install the `nginx` web server in Ubuntu 24.04, follow the guide on [How To Install Nginx on Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-22-04), including Step 5 – Setting Up Server Blocks (Recommended). It will help you set up a domain, which will be used very soon.

### Generating SSL certificates

To generate SSL certificates of your domain name for free, please go to [Cloudflare](https://cloudflare.com).

Click `Add Site` to add your domain name. They will give you two Cloudflare nameservers. Add these two nameservers as your `CustomDNS` on Namecheap by following [this guideline](https://www.namecheap.com/support/knowledgebase/article.aspx/767/10/how-to-change-dns-for-a-domain/).

Then on the page of your domain name on Cloudflare, click `DNS` and add the following two DNS records:

| Type | Name | Content |
| ---- | ---- | ------- |
|A     | your domain name | IPv4 address of your DigitalOcean Droplet |
|CNAME |   www            |your domain name      |

Finally, click `SSL/TLS` and then `Origin Server` to `Create Certificate`. Follow their steps to install a certificate. You should get an `Origin Certificate` and a `Private Key`.

Log in to your droplet and copy your `Origin Certificate` and `Private Key` to `/etc/nginx/ssl/<your domain name>.cer` and `/etc/nginx/ssl/<your domain name>.key`, respectively.

!!! success "Certificate Installation"
    After obtaining your `Origin Certificate` and `Private Key` from Cloudflare:

    1. Log in to your droplet
    2. Copy your `Origin Certificate` to `/etc/nginx/ssl/<your domain name>.cer`
    3. Copy your `Private Key` to `/etc/nginx/ssl/<your domain name>.key`

!!! warning "Important SSL/TLS Configuration"
    On Cloudflare, under `SSL/TLS`, please make sure you:

    1. Choose **`Full`** Encrypts end-to-end, using a self signed certificate on the server under `Overview`
    2. Check `Always Use HTTPS` under `Edge Certificates`

After all of the above-mentioned steps, enter your domain name into your browser's address bar, you should see a padlock symbol at the beginning of the address bar.

### Adjusting Your Nginx Server for Deploying Plato

Use the following configuration file in `/etc/nginx/sites-available/example.com` (where `example.com` is the domain name for the server):

```
server {
    listen      80;
    listen      443 ssl;
    server_name example.com www.example.com;
    root /home/username/example.com;
    index index.html index.htm index.php;

    ssl_certificate /etc/nginx/ssl/example.cer;
    ssl_certificate_key /etc/nginx/ssl/example.key;

    location / {
        try_files $uri $uri/ =404;
    }

    location /socket.io {
        proxy_pass http://127.0.0.1:8000/socket.io;
        proxy_redirect off;
        proxy_buffering off;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
    }

    location ~ /\.ht {
        deny all;
    }
}
```

This configuration file assumes that the certificate and private key for establishing an HTTPS connection to the server are stored as `/etc/nginx/ssl/example.cer` (your `Origin Certificate`) and `/etc/nginx/ssl/example.key` (your `Private Key`), respectively. It also assumes that all static content for the website is stored at `/home/username/example.com`.


If there is a need for using load balancing available from `nginx`, sticky sessions must be used for Socket.IO:

```
upstream example_servers {
    ip_hash; # enabling sticky sessions for Socket.IO
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
}

server {
    ...
    location /socket.io {
        proxy_pass http://example_servers;
        ...
    }
}
```

After the configuration file is modified, test it to make sure that there are no syntax errors:

```bash
$ sudo nginx -t
```

After seeing

```
nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
nginx: configuration file /etc/nginx/nginx.conf test is successful

```

restart the web server:

```
$ sudo systemctl restart nginx
```

### Installing Plato with uv

Clone the Plato repository to the desired directory on your server

```bash
$ git clone https://github.com/TL-System/plato.git
$ cd plato
```

Before using Plato, first install [uv](https://docs.astral.sh/uv/getting-started/installation/) with the commands below:

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ source $HOME/.local/bin/env
```

!!! warning "Troubleshooting uv"
    If it prompts `uv: command not found` when you enter any `uv` commands after successfully installing uv, use the command:

    ```bash
    $ source $HOME/.local/bin/env
    ```

Install Plato and its dependencies using uv:

```bash
$ uv sync
```

!!! warning "Installation Troubleshooting"
    If the installation gets killed when downloading packages, you can try:

    ```bash
    $ uv sync --no-cache
    ```

!!! tip "Time-Saving Alias"
    Use alias to save your time for running Plato in the future.

    ```bash
    $ vim ~/.bashrc
    ```

    Then add:

    ```bash
    alias plato='cd <directory of plato>/; source .venv/bin/activate'
    ```

    After saving this change and exiting:

    ```bash
    $ source ~/.bashrc
    ```

    Next time, after you SSH into this server, just type `plato` :)


### Starting Your Plato Server

The Plato federated learning server can be started as usual. For example:

```bash
$ cd examples/customized
$ uv run custom_server.py
```

On the side of a client, make sure that its configuration file contains the correct domain name of the production server deployed in the cloud:

```yaml
server:
    address: example.com
    use_https: true
```

!!! warning "Port Configuration"
    Do **NOT** specify `port: 8000` in a client's configuration file when deploying in production.

And then run the client as usual. For example:

```bash
$ uv run custom_client.py -i 1
```

There is no need to specify the port number for production servers deployed in the cloud.
