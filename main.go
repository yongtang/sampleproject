package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"os/exec"

	flatbuffers "github.com/google/flatbuffers/go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/encoding"

	"data/fbs"
)

type server struct{}

func (s *server) Record(context context.Context, in *fbs.Record) (*flatbuffers.Builder, error) {
	dataIn := in.Data()
	fmt.Fprintln(os.Stderr, string(dataIn))
	cmd := exec.Command("ls")
	pipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err = cmd.Start(); err != nil {
		return nil, err
	}
	reader := bufio.NewReader(pipe)
	dataOut, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	if err = cmd.Wait(); err != nil {
		return nil, err
	}
	b := flatbuffers.NewBuilder(0)
	i := b.CreateString(string(dataOut))
	fbs.RecordStart(b)
	fbs.RecordAddData(b, i)
	b.Finish(fbs.RecordEnd(b))
	return b, nil

}

func main() {
	flagClient := flag.String("client", "", "Client")
	flagServer := flag.String("server", "", "Server")
	flag.Parse()

	var clientAddress *string
	var serverAddress *string
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "client" {
			clientAddress = flagClient
		}
		if f.Name == "server" {
			serverAddress = flagServer
		}
	})
	if serverAddress != nil && clientAddress != nil {
		fmt.Fprintln(os.Stderr, "Please only specify either -client or -server")
		os.Exit(1)
	}
	if serverAddress == nil && clientAddress == nil {
		fmt.Fprintln(os.Stderr, "Please at least specify either -client or -server")
		os.Exit(1)
	}

	if clientAddress != nil {
		conn, err := grpc.Dial(*clientAddress, grpc.WithInsecure(), grpc.WithCodec(flatbuffers.FlatbuffersCodec{}))
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "Client: %s\n", *clientAddress)
		defer conn.Close()
		grpcClient := fbs.NewDataClient(conn)
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			b := flatbuffers.NewBuilder(0)
			i := b.CreateString(scanner.Text())
			fbs.RecordStart(b)
			fbs.RecordAddData(b, i)
			b.Finish(fbs.RecordEnd(b))
			r, err := grpcClient.Record(context.Background(), b)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			fmt.Println(string(r.Data()))
		}
	} else {
		listen, err := net.Listen("tcp", *serverAddress)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer listen.Close()
		grpcServer := grpc.NewServer(grpc.CustomCodec(flatbuffers.FlatbuffersCodec{}))
		defer grpcServer.GracefulStop()
		encoding.RegisterCodec(flatbuffers.FlatbuffersCodec{})
		fbs.RegisterDataServer(grpcServer, &server{})
		fmt.Fprintf(os.Stderr, "Server: %s\n", *serverAddress)
		if err = grpcServer.Serve(listen); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
}
