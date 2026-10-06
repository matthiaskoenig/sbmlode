# R with deSolve for the tests of the generated R code, see `rscript_command` of
# tests/ode_helpers.py:
#   docker build -t sbmlode-r -f tests/docker/r.Dockerfile tests/docker
#   SBMLODE_RSCRIPT="docker run --rm -v /tmp:/tmp sbmlode-r Rscript"
FROM rocker/r-ver:4.4
RUN Rscript -e 'install.packages("deSolve")'
# tini is the process 1 of the container and passes a signal on to Rscript, which as
# the process 1 itself would ignore the SIGTERM with which a test stops a hung job
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["/usr/bin/tini", "--"]
