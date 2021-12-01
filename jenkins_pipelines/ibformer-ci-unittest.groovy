/**
 * Run tests for every PR
 * and reports status back to Github
 */

// This is a workaround to abort already running builds on PR update.
// For more on milestones see: https://www.jenkins.io/doc/pipeline/steps/pipeline-milestone-step/
def buildNumber = env.BUILD_NUMBER as int
if (buildNumber > 1) milestone(buildNumber - 1)
milestone(buildNumber)

// check for env.CHANGE_FORK, if it's null (PR created from instabase/instabase instead of a fork)
// then default to "instabase"
def INSTABASE_FORK = env.CHANGE_FORK ?: 'instabase'

pipeline {
    agent { node { label 'build-machine' } }
    options {
        timeout(time: 120, unit: 'MINUTES')
    }
    stages {
        stage('Run CI tests on PR') {
            steps {
                dir ('.') {
                    ansiColor('xterm') {
                        sh '''#!/bin/bash
                        make build-docker
                        make run-docker-test
                        '''
                    }
                }
            }
        }
    }
    post {
        always {
            // TODO (oli): Remove sudo
            // using sudo till docker container used here to run tests stops using "root" user.
            // Due to bind mounts, some files are being owned by root inside the workspace
            // and git clean throws permission denied error.
            sh 'sudo git clean -fdx 2>/dev/null'
        }
        success {
            echo 'Success!'
        }
        failure {
            echo 'Failure!'
        }
    }
}