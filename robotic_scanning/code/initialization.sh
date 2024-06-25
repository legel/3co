# add following commands to .bashrc 

alias py="python3"

if [[ ! $(pgrep -f update_ip_addresses.py) ]]; then
    nohup python3 /home/pi/3cobot/update_ip_addresses.py >> /home/pi/3cobot/ip_logs.txt &
fi


if [[ ! $(pgrep -f motor_server.py) ]]; then
    cd 3cobot
    git pull
    nohup python3 /home/pi/3cobot/motor_server.py >> /home/pi/3cobot/logs.txt &
fi


if [[ ! $(pgrep -f busy_server.py) ]]; then
    nohup python3 /home/pi/3cobot/busy_server.py > /home/pi/3cobot/busy_logs.txt &
fi

# ssh key management
cp 3co ~/.ssh/3co
cp 3co.pub ~/.ssh/3co.pub
cp config ~/.ssh/config
chmod 400 ~/.ssh/3co
chmod 400 ~/.ssh/3co.pub

git config --global push.default matching
git remote set-url origin git@github.com:legel/3cobot.git
