# Samples

## Compile

Assume current directory is `/samples`

z/OS:

```
xlc  -g3 -qlanglvl=extc99 -Wc,LP64 -I ../zdnn -o simple_add simple_add.c ../zdnn/lib/libzdnn.x
```

Linux's:

```
gcc -g3 -Wall -fmessage-length=0 -std=c99 -I ../zdnn -o simple_add simple_add.c  ../zdnn/lib/libzdnn.so
```

### NOTE: Add `-D STATIC_LIB` to gcc invocation if you're compiling using statically-linked library
