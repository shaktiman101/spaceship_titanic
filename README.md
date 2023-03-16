# spaceship_titanic
Kaggle Spaceship Titanic project - "Predict which passengers are transported to an alternate dimension"

# Steps to run the repo
1. Clone the repo
2. cd to the repo, 'spaceship_titanic'
3. Run command to build docker image: docker build -t spaceship_titanic
4. Run command to run the docker container: docker run -it --name spaceship_titanic_service -v %cd%:/app spaceship_titanic
