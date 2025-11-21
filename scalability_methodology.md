# Scalability Testing Methodology

## Test Setup
- **Broker**: Mosquitto 2.0.15 on Ubuntu 22.04
- **Hardware**: 8-core Intel i7, 16GB RAM
- **Clients**: Python asyncio with 50 coroutines per process
- **Processes**: 10 processes × 50 coroutines = 500 concurrent clients
- **Message rate**: 1 msg/sec per client
- **Duration**: 60 seconds sustained load

## Resource Limits
- CPU usage: <80% on broker
- Memory: <4GB for broker process
- Network: 1Gbps local network
- File descriptors: ulimit -n 65536

## Measurements
- Connection success rate
- Message delivery ratio under load
- 95th percentile latency
