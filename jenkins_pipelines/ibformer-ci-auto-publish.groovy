def BRANCH = params.RELEASE_BRANCH.toLowerCase()
addShortText background: 'white', border: 0, borderColor: 'white', color: 'black', link: '', text: "${BRANCH}"

// @NonCPS
def getChangeString() {
    MAX_MSG_LEN = 100
    def changeString = ''
    def changeLogSets = currentBuild.changeSets
    for (int i = 0; i < changeLogSets.size(); i++) {
        def entries = changeLogSets[i].items
        for (int j = 0; j < entries.length; j++) {
            def entry = entries[j]
            truncated_msg = entry.msg.take(MAX_MSG_LEN)
            changeString += " - ${truncated_msg} [${entry.author}]\n"
        }
    }
    if (!changeString) {
        changeString = ' - No new changes'
    }
    return changeString
}


def postResultsToSlack(environment, color) {
    env.LAST_GIT_COMMIT = sh(script: "git log -1 --pretty=format:'[%h] (%an) %s'", returnStdout: true).trim()
    env.CHANGE_LOG = getChangeString()

    if (INSTABASE_FORK == "instabase" && RELEASE_BRANCH == "main") {
        def slackResponseBad = slackSend(
            botUser: true,
            channel: '#alerts-ibformer-ci-tests',
            color: color,
            message: "${currentBuild.fullDisplayName} from ${RELEASE_BRANCH} was ${currentBuild.currentResult.toLowerCase()} in environment ${environment}! See ${BUILD_URL}console for more information.",
        )

        slackSend(
            botUser: true,
            channel: slackResponseBad.threadId,
            message: "Latest Commit: ```${LAST_GIT_COMMIT}```\n\nChange Log:\n```${CHANGE_LOG}```\n",
        )
    }
}


pipeline {
    agent { node { label 'build-machine' } }
    parameters {
        string(
            name: 'RELEASE_BRANCH',
            defaultValue: 'main',
            description: 'Specify the git branch'
        )
        string(
            name: 'INSTABASE_FORK',
            defaultValue: 'instabase',
            description: 'Specify the git fork'
        )
    }
    environment {
        PATH = '/nodenv/bin:/nodenv/shims:/home/ib-build/.pyenv/shims:/home/ib-build/.pyenv/bin:/usr/local/openjdk-8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin:/usr/local/workspace/go/bin'
    }
    options {
        timeout(time: 50, unit: 'MINUTES')
    }
    stages {
        stage ('Clone repo') {
            steps {
                git (
                    branch: "${RELEASE_BRANCH}",
                    credentialsId: 'ib-build-sshkey',
                    url: "git@github.com:${INSTABASE_FORK}/ibformers.git",
                )
            }
        }
        stage ('Build docker contianer for CI testing') {
            steps {
                dir ('.') {
                    ansiColor('xterm') {
                        sh '''#!/bin/bash
                        cd ci/
                        make build-docker
                        '''
                    }
                }
            }
        }
        stage ('Run ibformer ci tests in each environment:') {
            parallel {
                stage ('[Run ibformer tests on doc-insights-sandbox and publish if run successfully]') {
                    steps {
                        dir ('.') {
                            ansiColor('xterm') {
                                sh '''#!/bin/bash
                                cd ci/
                                IB_TEST_ENV=doc-insights-sandbox make run-docker-test
                                '''
                            }
                        }
                    }
                    post {
                        success {
                            dir ('.') {
                                ansiColor('xterm') {
                                    sh '''#!/bin/bash
                                    cd ci/
                                    IB_TEST_ENV=doc-insights-sandbox make run-docker-publish
                                    '''
                                }
                            }
                            postResultsToSlack('doc-insights-sandbox', 'good')
                        }
                        unsuccessful {
                            postResultsToSlack('doc-insights-sandbox', 'danger')
                        }
                    }
                }
                stage ('[Run ibformer tests on dogfood and publish if run successfully]') {
                    steps {
                        dir ('.') {
                            ansiColor('xterm') {
                                sh '''#!/bin/bash
                                cd ci/
                                IB_TEST_ENV=dogfood make run-docker-test
                                '''
                            }
                        }
                    }
                    post {
                        success {
                            dir ('.') {
                                ansiColor('xterm') {
                                    sh '''#!/bin/bash
                                    cd ci/
                                    IB_TEST_ENV=dogfood make run-docker-publish
                                    '''
                                }
                            }
                            postResultsToSlack('dogfood', 'good')
                        }
                        unsuccessful {
                            postResultsToSlack('dogfood', 'danger')
                        }
                    }
                }
                stage ('[Run ibformer tests on prod and publish if run successfully]') {
                    steps {
                        dir ('.') {
                            ansiColor('xterm') {
                                sh '''#!/bin/bash
                                cd ci/
                                IB_TEST_ENV=prod make run-docker-test
                                '''
                            }
                        }
                    }
                    post {
                        success {
                            dir ('.') {
                                ansiColor('xterm') {
                                    sh '''#!/bin/bash
                                    cd ci/
                                    IB_TEST_ENV=prod make run-docker-publish
                                    '''
                                }
                            }
                            postResultsToSlack('prod', 'good')
                        }
                        unsuccessful {
                            postResultsToSlack('prod', 'danger')
                        }
                    }
                }
            }
        }
    }
    post {
        cleanup {
            sh 'sudo git clean -ffdx'
        }
    }
}