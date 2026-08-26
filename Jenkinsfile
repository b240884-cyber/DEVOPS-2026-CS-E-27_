pipeline {
    agent any

    environment {
        PYTHONPATH = "src"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python Environment') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            python3 -m venv .venv
                            .venv/bin/pip install --upgrade pip
                        '''
                    } else {
                        bat '''
                            python -m venv .venv
                            .venv\\Scripts\\pip install --upgrade pip
                        '''
                    }
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                script {
                    if (isUnix()) {
                        sh '.venv/bin/pip install -r requirements.txt ruff'
                    } else {
                        bat '.venv\\Scripts\\pip install -r requirements.txt ruff'
                    }
                }
            }
        }

        stage('Lint with Ruff') {
            steps {
                script {
                    if (isUnix()) {
                        sh '.venv/bin/ruff check .'
                    } else {
                        bat '.venv\\Scripts\\ruff check .'
                    }
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    if (isUnix()) {
                        sh '.venv/bin/python src/train.py'
                    } else {
                        bat '.venv\\Scripts\\python src/train.py'
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Pipeline execution complete.'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Please check build logs.'
        }
    }
}
