![PyGrid logo](https://raw.githubusercontent.com/OpenMined/design-assets/master/logos/PyGrid/horizontal-primary-trans.png)


PyGrid Node is a server based application used by the [PyGrid Platform](https://github.com/OpenMined/PyGrid/).


## Overview
- [Start Grid Nodes Locally](#start-grid-nodes-locally)
- [Try out the Tutorials](#try-out-the-tutorials)
- [Start Contributing](#start-contributing)
- [Disclaimer](#disclaimer)
- [License](#license)


### Start Grid Nodes locally


#### Using Python
To start a grid node using python, run:
```
python grid_node.py 
```
You can pass the arguments or use environment variables to set the grid node configs.  

**Arguments**
```
  -h, --help                shows the help message and exit
  --id [ID]                 the grid node identifier, e.g. --id=alice.
  -p [PORT], --port [PORT]  port to run the server on
  --host [HOST]             the grid node host
  --gateway_url [URL]       address used to join a Grid Network.
  --db_url [URL]            REDIS database server address
```

**Environment Variables**
- `ID` - The grid node identifier
- `PORT` -  Port to run server on.
- `ADDRESS` - The grid node address/host
- `REDISCLOUD_URL` - The redis database URL
- `GRID_GATEWAy_URL` - The address used to join a Grid Network

### Docker

The latest PyGrid Node image are available on the Docker Hub  

PyGrid Node Docker image - `openmined/grid-node`

#### Pulling images
```
$ docker pull openmined/grid-node  # Download grid node image
```

#### Build your own PyGrid Node image
```
$ docker build openmined/grid-node . # Build grid node image
```

## Try out the Tutorials
A comprehensive list of tutorials can be found [here](https://github.com/OpenMined/PySyft/tree/master/examples/tutorials/grid).

These tutorials cover how to create a PyGrid node and what operations you can perform.

## Start Contributing
The guide for contributors can be found [here](https://github.com/OpenMined/PyGrid/tree/dev/CONTRIBUTING.md). It covers all that you need to know to start contributing code to PyGrid in an easy way.

Also join the rapidly growing community of 7300+ on [Slack](http://slack.openmined.org). The slack community is very friendly and great about quickly answering questions about the use and development of PyGrid/PySyft!

We also have a Github Project page for a Federated Learning MVP [here](https://github.com/orgs/OpenMined/projects/13).  
You can check the PyGrid's official development and community roadmap [here](https://github.com/OpenMined/Roadmap/tree/master/pygrid_team).


## Disclaimer
Do ***NOT*** use this code to protect data (private or otherwise) - at present it is very insecure.

## License

[Apache License 2.0](https://github.com/OpenMined/PyGrid/blob/dev/LICENSE)
