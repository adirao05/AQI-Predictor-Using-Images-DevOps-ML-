pipeline {
    agent any

    environment {
        DOCKER_USER = "adirao05"
        IMAGE_NAME  = "${DOCKER_USER}/aqi-devops-app"
    }

    stages {

        stage('Checkout') {
            steps {
                git branch: 'main',
                    credentialsId: 'github-creds',
                    url: 'https://github.com/adirao05/AQI-Predictor-Using-Images-DevOps-ML-'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat "docker build -t ${IMAGE_NAME}:latest ."
            }
        }

        stage('Push to Docker Hub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DUSER',
                    passwordVariable: 'DPASS'
                )]) {
                    bat "docker login -u %DUSER% -p %DPASS%"
                    bat "docker push ${IMAGE_NAME}:latest"
                }
            }
        }

        stage('Run Container') {
            steps {
                bat "docker stop aqi-app || exit 0"
                bat "docker rm aqi-app || exit 0"
                bat "docker run -d --name aqi-app -p 8501:8501 ${IMAGE_NAME}:latest"
            }
        }
    }

    post {
        success { echo '✅ AQI Pipeline completed!' }
        failure { echo '❌ Pipeline failed. Check logs.' }
    }
}