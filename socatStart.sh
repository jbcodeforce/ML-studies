
socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"host.docker.internal:0\" 
