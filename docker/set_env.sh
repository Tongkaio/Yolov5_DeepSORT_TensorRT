#!/bin/bash

# 1. 创建 .scripts 目录并下载 git-prompt.sh 和 git-completion.bash
mkdir -p ~/.scripts
curl -L https://raw.github.com/git/git/master/contrib/completion/git-prompt.sh -o ~/.scripts/git-prompt.sh
curl -L https://raw.github.com/git/git/master/contrib/completion/git-completion.bash -o ~/.scripts/git-completion.bash

# 2. 设置 locale
locale-gen zh_CN.UTF-8
locale-gen zh_SG.UTF-8
update-locale LANG=zh_CN.UTF-8

# 3. 更新 .bashrc 文件
echo 'export LANG=zh_CN.UTF-8' >> ~/.bashrc
echo 'export LANGUAGE=zh_CN:zh' >> ~/.bashrc
echo 'export LC_ALL=zh_CN.UTF-8' >> ~/.bashrc
echo 'source ~/.scripts/git-prompt.sh' >> ~/.bashrc
echo 'source ~/.scripts/git-completion.bash' >> ~/.bashrc
echo 'export PS1="\[\e[1;33m\]λ\[\e[0m\] \h \[\e[1;32m\]\w\[\e[1;33m\]\$(__git_ps1 \" \[\e[35m\]{\[\e[36m\]%s\[\e[35m\]}\") \[\e[0m\]"' >> ~/.bashrc

# 4. 使 .bashrc 更改生效
source ~/.bashrc
