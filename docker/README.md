CS 286 Docker
=============

The [Docker][] container-based virtualization service lets you run a minimal
Linux environment on a Mac OS X or Windows computer, without the overhead of a
full virtual machine like [VMware Workstation][], [VMware Fusion][], or
[VirtualBox][].

Advantages of Docker:

* Docker can start and stop virtual machines incredibly quickly.
* Docker-based virtual machines are small and occupy little space on your machine.
* With Docker, you can easily *edit* your code in your home environment, but
  *compile and run* it on a Linux host.

Disadvantages of Docker:

* Docker does not offer a graphical environment. You will need to run commands
  exclusively in the terminal.
* Docker technology is less user-friendly than virtual machines. You'll have
  to type weird commands.
* You won't get the fun, different feeling of a graphical Linux desktop.


## What's in this image

The image is based on `ubuntu:questing` and includes:

* **C++ toolchain**: GCC 15, Clang 21, GDB, CMake — for building the C++
  cotamer simulation (`robot_sim/`)
* **Python 3 + PyTorch (CPU)** — for running the MAPPO training script
  (`train.py`) and the `gaussian_field_env.py` environment
* **Networking tools**: iproute2, netcat, tcpdump, etc.
* The same base packages as the CS 2620 image (editors, man pages, sudo, etc.)


## Building the image

1. Download and install [Docker][].

2. Clone this repository onto your computer.

3. Change into the `docker/` directory.

4. Run:

    ```shellsession
    $ ./build-docker
    ```

    This builds a snapshot named `cs286:latest`. It will take a while the
    first time (downloading PyTorch adds a few minutes); subsequent builds
    reuse cached layers and are much faster.

> If `./build-docker` fails with packages missing or out of date, try
> `./build-docker --no-cache` instead.

## Running the container

### By hand

Mount the repo root into the container so your edits are immediately visible:

```shellsession
$ docker run -it --rm -v ~/path/to/multi-lvl-comms:/home/cs286-user/multi-lvl-comms cs286:latest
```

Then inside the container:

```shellsession
cs286-user@a47f05ea5085:~$ cd multi-lvl-comms
cs286-user@a47f05ea5085:~/multi-lvl-comms$ python3 train.py --episodes 5
ep    0 | R=  64.45 | wm=8.6996  v=57.4526  π=-0.0000
...
cs286-user@a47f05ea5085:~/multi-lvl-comms$ cd robot_sim && make
...
cs286-user@a47f05ea5085:~/multi-lvl-comms$ cs61-docker-version
25-cs286
cs286-user@a47f05ea5085:~/multi-lvl-comms$ exit
exit
$
```

A prompt like `cs286-user@a47f05ea5085:~$` means your terminal is connected to
the Linux VM. Type Control-D or `exit` to leave.

### Flags explained

* `docker run` — start a new container
* `-it` — interactive (`-i`) with a terminal (`-t`)
* `--rm` — remove the container when it exits
* `-v LOCALDIR:LINUXDIR` — share a directory between host and container;
  edits on either side are immediately visible to the other
* `cs286:latest` — the image built by `./build-docker`

### Using a custom tag

```shellsession
$ ./build-docker --tag myname:latest
$ docker run -it --rm -v ~/multi-lvl-comms:/home/cs286-user/multi-lvl-comms myname:latest
```

[Docker]: https://docker.com/
[VMware Workstation]: https://www.vmware.com/products/workstation-player.html
[VMware Fusion]: https://www.vmware.com/products/fusion.html
[VirtualBox]: https://www.virtualbox.org/
