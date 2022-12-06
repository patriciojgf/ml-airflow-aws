#conectar al bastion
ssh -i "bastion_host_ml.pem" ec2-user@ec2-18-212-217-123.compute-1.amazonaws.com
ssh -i "bastion_host_ml.pem" ubuntu@ec2-3-238-138-98.compute-1.amazonaws.com
ServerAliveInterval 50


git clone https://github.com/patriciojgf/ml-airflow-aws.git


### Docker and docker compose prerequisites
sudo apt-get install curl
sudo apt-get install gnupg
sudo apt-get install ca-certificates
sudo apt-get install lsb-release

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

### Add Docker and docker compose support to the Ubuntu's packages list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-pluginsudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-pluginlinux/ubuntu   $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null


sudo apt-get update
 
### Install docker and docker compose on Ubuntu
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
 
sudo apt  install docker-compose


sudo docker-compose up -d