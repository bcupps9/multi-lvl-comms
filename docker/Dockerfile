FROM ubuntu:questing

# set environment variables for tzdata
ARG TZ=America/New_York
ENV TZ=${TZ}

# include manual pages and documentation
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update &&\
 apt-get -y install unminimize &&\
 yes | unminimize

# install GCC-related packages
RUN apt-get -y install\
 binutils-doc\
 cpp-doc\
 gcc-doc\
 g++\
 gdb\
 gdb-doc\
 glibc-doc\
 libblas-dev\
 liblapack-dev\
 liblapack-doc\
 libstdc++-15-doc\
 libstdc++-15-pic\
 libstdc++6-15-dbg\
 make\
 make-doc

# install clang-related packages
RUN apt-get -y install\
 clang-21\
 clang-21-doc\
 lldb-21

# install programs used for system exploration
RUN apt-get -y install\
 strace\
 tcpdump

# install interactive programs (emacs, vim, nano, man, sudo, etc.)
RUN apt-get -y install\
 bc\
 curl\
 dc\
 emacs-nox\
 git\
 git-doc\
 man\
 micro\
 nano\
 psmisc\
 sudo\
 vim\
 wget

# set up libraries
RUN apt-get -y install\
 libjson-perl\
 libreadline-dev\
 locales\
 wamerican

# install programs used for networking
RUN apt-get -y install\
 dnsutils\
 inetutils-ping\
 iproute2\
 net-tools\
 netcat-openbsd\
 telnet\
 time\
 traceroute

# install programs useful for disk images
RUN apt-get -y install\
 fdisk\
 mtools

# install libraries and programs needed for CS 2620 psets
RUN apt-get -y install\
 cmake\
 libssl-dev\
 openssl\
 libxxhash-dev\
 protobuf-compiler\
 libprotobuf-dev\
 libgrpc-dev\
 libgrpc++-dev\
 protobuf-compiler-grpc

# install libraries useful for optimization and performance analysis
RUN apt-get -y install\
    linux-tools-common\
    linux-tools-generic

# install Python and pip for the SeqComm training stack
RUN apt-get update &&\
 apt-get -y install\
 python3\
 python3-pip\
 python3-venv

# install Python deps from the repo's requirements.txt
# CPU-only torch wheel — no CUDA needed inside the container
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages \
    --index-url https://download.pytorch.org/whl/cpu \
    -r /tmp/requirements.txt

# set up default locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV EDITOR=nano

# remove unneeded .deb files
# RUN rm -r /var/lib/apt/lists/*

# set up passwordless sudo for user cs286-user
ARG UID=1001
RUN useradd -m -s /bin/bash -u "${UID}" cs286-user &&\
 echo "cs286-user ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/cs286-init

# create binary reporting version of dockerfile
RUN (echo '#\!/bin/sh'; echo 'echo 25-cs286') > /usr/bin/cs61-docker-version; chmod ugo+rx,u+w,go-w /usr/bin/cs61-docker-version

# git build arguments
ARG USER=CS61\ User
ARG EMAIL=nobody@example.com

# ensure /usr/bin/c++, /usr/bin/g++, /usr/bin/clang++
RUN test -z "`which c++`" && ln -s g++-15 /usr/bin/c++;\
 test -z "`which g++`" && ln -s g++-15 /usr/bin/g++;\
 test -z "`which clang`" && ln -s clang-21 /usr/bin/clang;\
 test -z "`which clang++`" && ln -s clang++-21 /usr/bin/clang++;\
 test -z "`which lldb`" && ln -s lldb-21 /usr/bin/lldb

# configure your environment
USER cs286-user
RUN git config --global user.name "${USER}" &&\
 git config --global user.email "${EMAIL}" &&\
 git config --global safe.directory "*" &&\
 (echo "(custom-set-variables"; echo " '(c-basic-offset 4)"; echo " '(indent-tabs-mode nil))") > ~/.emacs &&\
 (echo "set expandtab"; echo "set shiftwidth=4"; echo "set softtabstop=4") > ~/.vimrc &&\
 cat /dev/null > ~/.bash_profile &&\
 echo "# 2022: avoid a Docker bug with user mapping by listing working directory" >> ~/.bash_profile &&\
 echo "ls -al > /dev/null" >> ~/.bash_profile &&\
 echo "for i in \`mount | grep /home/cs286-user | sed 's/^.*\\(\\/home[^ ]*\\).*/\\\\1/'\`; do ls -al \$i > /dev/null; done" >> ~/.bash_profile &&\
 echo "# make ssh-auth.sock user-readable" >> ~/.bash_profile &&\
 (echo "if test -e /run/host-services/ssh-auth.sock; then"; echo "  sudo chown cs286-user:cs286-user /run/host-services/ssh-auth.sock"; echo "fi") >> ~/.bash_profile &&\
 echo ". ~/.bashrc" >> ~/.bash_profile &&\
 rm -f ~/.bash_logout &&\
 echo "add-auto-load-safe-path ~" > ~/.gdbinit

WORKDIR /home/cs286-user
CMD ["/bin/bash", "-l"]

# Initial version of this Dockerfile by Todd Morrill, CS 61 DCE student
