# Deploying a Plato Federated Learning Server in a Production Environment

The Plato federated learning server is designed to use Socket.IO over HTTP and HTTPS, and can be easily deployed in a production server environment in the public cloud. 

To deploy such a production federated learning server in a virtual machine from any public cloud provider, a `nginx` web server will first need to be installed to serve as a reverse proxy server. To install the `nginx` web server in Ubuntu 20.04, follow Step 1 in the guide on [How To Install Linux, Nginx, MySQL, PHP (LEMP stack) on Ubuntu 20.04](https://www.digitalocean.com/community/tutorials/how-to-install-linux-nginx-mysql-php-lemp-stack-on-ubuntu-20-04).

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

After the configuration file is created, create a symbolic link in `/etc/nginx/sites-enabled`:

```shell
sudo ln -s /etc/nginx/sites-available/example.com /etc/nginx/sites-enabled/
```

Then test and restart the web server:

```shell
sudo nginx -t
sudo systemctl restart nginx
```

The Plato federated learning server can be started as usual. For example:

```shell
cd examples
python custom_server.py
```

On the clients, make sure that the configuration file contains the correct domain name of the production server deployed in the cloud:

```
server:
    address: example.com
    use_https: true
```

And then run the clients as usual. For example:
```shell
python custom_client.py -i 1
```

There is no need to specify the port number for production servers deployed in the cloud.
