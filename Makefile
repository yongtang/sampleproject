.PHONY: all
all:
	CGO_ENABLED=0 go build -v -o go.exe .

.PHONY: gen
gen:
	docker run -i --rm -v $$PWD:/v -w /v ubuntu:20.04 bash -c 'apt-get -y -qq update && apt-get -y -qq install flatbuffers-compiler && flatc --go --grpc -o . fbs/schema.fbs'

.PHONY: clean
clean:
	go clean
	rm -f go.exe
