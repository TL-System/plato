# Deploying a Plato Federated Learning Server with DigitalOcean

This is a more detailed documentation of deploying a Plato federated learning server in one of the production environments -- DigitalOcean. 

## Prerequisites

### Creating your droplet

First thing first, create an account on [DigitalOcean](https://www.digitalocean.com) if you haven't, and sign in.

For your future convenience, follow [this tutorial](https://docs.digitalocean.com/products/droplets/how-to/add-ssh-keys/) to use SSH keys with your **Droplets** (DigitalOcean Droplets are Linux-based virtual machines that run on top of virtualized hardware. Each Droplet you create is a new server you can use.)

Then go back to your DigitalOcean homepage (control panel), click the green button `Create` on the upper right corner and choose `Droplets`.  Here we need to create a droplet to use it as your Plato federated learning server. Choose Ubuntu 20.04 (LTS) x64 image. 

**Note**: Don't forget to check `IPv6` in the `Select additional options` to enable public IPv6 networking, so that you can SSH to your server right after you create it.

After creating your droplet, click `Droplets` under `MANAGE` on the left-hand side of your control panel, you will see the name of the droplet you just create. Click it and you will see all the information of it, including its IP address. 

Open terminal on your local machine and you should be able to log in to your droplet as the `root` user:

```
$ ssh root@<IP address of your droplet>
```

Then configure a regular user account by following [this guide](https://www.digitalocean.com/community/tutorials/initial-server-setup-with-ubuntu-20-04).


### Purchasing a domain name

Purchase a domain name that will be used later. A recommended place to purchase it is [Namecheap](namecheap.com), where you can easily search and buy affordable domain names.



## Installing Nginx

To deploy a production federated learning server in a virtual machine from any public cloud provider, a `nginx` web server will first need to be installed to serve as a reverse proxy server. 

To install the `nginx` web server in Ubuntu 20.04, follow the guide on [How To Install Nginx on Ubuntu 20.04](https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-20-04), including Step 5 – Setting Up Server Blocks (Recommended). It will help you set up a domain, which will be used very soon. 


## Generating SSL certificates 

To generate SSL certificates of your domain name for free, please go to [Cloudflare](cloudflare.com). 

Click `Add Site` to add your domain name. They will give you two Cloudflare nameservers. Add these two nameservers as your `CustomDNS` on Namecheap by following [this guideline](https://www.namecheap.com/support/knowledgebase/article.aspx/767/10/how-to-change-dns-for-a-domain/).

Then on the page of your domain name on Cloudflare, click `DNS` and add the following two DNS records:

| Type | Name | Content |
| ---- | ---- | ------- |
|A     | your domain name | IPv4 address of your DigitalOcean Droplet |
|CNAME |   www            |your domain name      |

Finally, click `SSL/TLS` and then `Origin Server` to `Create Certificate`. Follow their steps to install a certificate. You should get an `Origin Certificate` and a `Private Key`. 

Log in to your droplet and copy your `Origin Certificate` and `Private Key` to `/etc/nginx/ssl/<your domain name>.cer` and `/etc/nginx/ssl/<your domain name>.key`, respectively.

**Note**: On Cloudflare, under `SSL/TLS`, please make sure you 

1. choose **`Full`** Encrypts end-to-end, using a self signed certificate on the server under `Overview`;
2. check `Always Use HTTPS` under `Edge Certificates`.

After all of the above-mentioned steps, enter your domain name into your browser's address bar, you should see a padlock symbol at the beginning of the address bar.



## Adjusting Your Nginx Server for Deploying Plato

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

```shell
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

## Installing Plato with PyTorch

Clone the Plato repository to the desired directory on your server

```
$ git clone https://github.com/TL-System/plato.git
```

Before using Plato, first install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with the commands below:

```
$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```

If it prompts `conda: command not found` when you enter any `conda` commands after successfully installing Miniconda, use the command:

```
$ source ~/.bashrc
```

Update your `conda` environment, and then create a new `conda` environment with Python 3.8 using the command:

```shell
$ conda update conda
$ conda create -n federated python=3.8
$ conda activate federated
```

The next step is to install the required Python packages. PyTorch should be installed following the advice of its [getting started website](https://pytorch.org/get-started/locally/). Since we use Ubuntu without CUDA GPU support, the command to install PyTorch would be:

```
$ conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Then install Plato as a pip package (make sure you are under the plato directory before doing this):

```
$ pip install .
```

If it gets killed when downloading packages, try

```
pip install . --no-cache-dir
```

**Tip:** Use alias to save your time for running Plato in the future.

```
$ vim ~/.bashrc
```

Then add 

```
alias plato='cd <directory of plato>/; conda activate federated'
```

After saving this change and exiting, 

```
$ source ~/.bashrc
```

Next time, after you SSH into this server, just type `plato`:)


## Starting Your Plato Server

The Plato federated learning server can be started as usual. For example:

```shell
$ cd examples/customized
$ python custom_server.py
```

On the side of a client, make sure that its configuration file contains the correct domain name of the production server deployed in the cloud:

```
server:
    address: example.com
    use_https: true
```

**Note**: Do **NOT** specify `port: 8000` in a client's configuration file. 

And then run the client as usual. For example:

```shell
$ python custom_client.py -i 1
```

There is no need to specify the port number for production servers deployed in the cloud.
