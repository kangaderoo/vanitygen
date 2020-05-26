LIBS=-lpcre -lcrypto -lm -lpthread
CFLAGS=-std=c99 -ggdb -O3 -Wall -march=native
OBJS=vanitygen.o oclvanitygen.o oclvanityminer.o oclengine.o keyconv.o pattern.o util.o rmd160.o sha256.o custom_ec_bn.o winglue.o json.o segwit_addr.o
#PROGS=vanitygen keyconv oclvanitygen oclvanityminer
PROGS=vanitygen 

#PLATFORM=$(shell uname -s)
#ifeq ($(PLATFORM),Darwin)
#OPENCL_LIBS=-framework OpenCL
#else
#OPENCL_LIBS=-lOpenCL
#endif
OPENCL_LIBS=-lopenssl


most: vanitygen keyconv

all: $(PROGS)

vanitygen: vanitygen.o pattern.o util.o rmd160.o sha256.o custom_ec_bn.o winglue.o json.o segwit_addr.o
	$(CC) $^ -o $@ $(CFLAGS) $(LIBS)

oclvanitygen: oclvanitygen.o oclengine.o pattern.o util.o winglue.o json.o segwit_addr.o
	$(CC) $^ -o $@ $(CFLAGS) $(LIBS) $(OPENCL_LIBS)

oclvanityminer: oclvanityminer.o oclengine.o pattern.o util.o winglue.o json.o segwit_addr.o
	$(CC) $^ -o $@ $(CFLAGS) $(LIBS) $(OPENCL_LIBS) -lcurl

keyconv: keyconv.o util.o winglue.o json.o segwit_addr.o
	$(CC) $^ -o $@ $(CFLAGS) $(LIBS)

clean:
	rm -f $(OBJS) $(PROGS) $(TESTS)
