import park
from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.utils.directed_graph import DirectedGraph
import subprocess as sp
import time
import zmq
import os
import signal
import numpy as np
import json
import networkx as nx
import psutil
import pdb
import signal

def find_available_port(orig_port):
    conns = psutil.net_connections()
    ports = [c.laddr.port for c in conns]
    new_port = orig_port

    while new_port in ports:
        new_port += 1
    return new_port

class QueryOptEnv(core.Env):
    """
    TODO: describe state, including the features for nodes and edges.
    """
    def __init__(self):
        signal.signal(signal.SIGINT, self._handle_exit_signal)
        signal.signal(signal.SIGTERM, self._handle_exit_signal)
        self.base_dir = None    # will be set by _install_dependencies
        self._install_dependencies()
        # original port:
        self.port = find_available_port(config.qopt_port)

        # start calcite + java server
        logger.info("port = " + str(self.port))
        self._start_java_server()

        context = zmq.Context()
        #  Socket to talk to server
        logger.info("Going to connect to calcite server")
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://localhost:" + str(self.port))
        self.reward_normalization = config.qopt_reward_normalization

        # TODO: describe spaces
        self.graph = None
        self.action_space = None
        # here, we specify the edge to choose next to calcite using the
        # position in the edge array
        # this will be updated every time we use _observe
        self._edge_pos_map = None

        self.query_set = self._send("getCurQuerySet")
        self.attr_count = int(self._send("getAttrCount"))

        # FIXME: make this nicer?
        # set dependent flags together:
        # if config.qopt_runtime_reward:
            # config.qopt_only_final_reward = 1
            # config.qopt_reward_normalization = ""

        # FIXME: these variable don't neccessarily belong here / should be
        # cleaned up
        # TODO: figure this out using the protocol too. Or set it on the java
        # side using some protocol.
        # self.only_final_reward = config.qopt_only_final_reward

        # will store _min_reward / _max_reward for each unique query
        # will map query: (_min_reward, _max_reward)
        self.reward_mapper = {}
        # these values will get updated in reset.
        self._min_reward = None
        self._max_reward = None
        self.current_query = None

        # setup space with the new graph
        self._setup_space()

        # more experimental stuff

        # original graph used to denote state
        self.orig_graph = None
        if config.qopt_viz:
            self.viz_ep = 0
            self.viz_output_dir = "./visualization/"
            self.viz_title_tmp = "query: {query}, ep: {ep}, step: {step}"

    def _install_if_needed(self, name):
        '''
        checks if the program called `name` is installed on the local system,
        and prints an error and cleans up if it isn't.
        '''
        # Check if docker is installed.
        cmd_string = "which {}".format(name)
        # which_pg_output = sp.check_output(cmd_string.split())
        process = sp.Popen(cmd_string.split())
        ret_code = process.wait()
        if ret_code != 0:
            # TODO: different installations based on OS, so this should be
            # user's responsibility
            print("{} is not installed. Please install docker before \
                    proceeding".format(name))
            env.clean()

    def _install_dependencies(self):
        """
         - clone OR pull latest version from query-optimizer github repo
            - TODO: auth?
            - set path appropriately to run java stuff
         - use docker to install, and start postgres
        """
        logger.info("installing dependencies for query optimizer")
        self.base_dir = park.__path__[0]
        self._install_if_needed("docker")
        self._install_if_needed("mvn")
        self._install_if_needed("java")

        # set up the query_optimizer repo
        try:
            qopt_path = os.environ["QUERY_OPT_PATH"]
        except:
            # if it has not been set, then set it based on the base dir
            qopt_path = self.base_dir + "/query-optimizer"
            # if this doesn't exist, then git clone this
            if not os.path.exists(qopt_path):
                print("going to clone query-optimizer library")
                cmd = "git clone https://github.com/parimarjan/query-optimizer.git"
                p = sp.Popen(cmd, shell=True,
                    cwd=self.base_dir)
                p.wait()
                print("cloned query-optimizer library")
                # now need to install the join-order-benchmark as well
                JOB_REPO = "https://github.com/gregrahn/join-order-benchmark.git"
                cmd = "git clone " + JOB_REPO
                p = sp.Popen(cmd, shell=True,
                    cwd=qopt_path)
                p.wait()
                print("downloaded join order benchmark queries")
        print("query optimizer path is: ", qopt_path)

        # TODO: if psql -d imdb already set up locally, then do not use docker
        # to set up postgres. Is this really useful, or should we just assume
        # docker is always the way to go?

        # TODO: print plenty of warning messages: going to start docker,
        # docker's directory should have enough space - /var/lib/docker OR
        # change it manually following instructions at >>>> .....

        docker_dir = qopt_path + "/docker"
        docker_img_name = "pg"
        container_name = "docker-pg"
        # docker build
        docker_bld = "sudo docker build -t {} . ".format(docker_img_name)
        p = sp.Popen(docker_bld, shell=True, cwd=docker_dir)
        p.wait()
        print("building docker image {} successful".format(docker_img_name))
        time.sleep(2)
        # start / or create new docker container
        # Note: we need to start docker in a privileged mode so we can clear
        # cache later on.
        docker_run = "sudo docker run --name {} -p \
        5400:5432 --privileged -d {}".format(container_name, docker_img_name)
        docker_start_cmd = "sudo docker start docker-pg || " + docker_run
        p = sp.Popen(docker_start_cmd, shell=True, cwd=docker_dir)
        p.wait()
        print("starting docker container {} successful".format(container_name))
        time.sleep(2)

        check_container_cmd = "sudo docker ps | grep {}".format(container_name)
        process = sp.Popen(check_container_cmd, shell=True)
        ret_code = process.wait()
        if ret_code != 0:
            print("something bad happened when we tried to start docker container")
            print("got ret code: ", ret_code)
            env.clean()

        time.sleep(2)
        # need to ensure that we psql has started in the container. If this is
        # the first time it is starting, then pg_restore could take a while.
        import psycopg2
        while True:
            try:
                conn = psycopg2.connect(host="localhost",port=5400,dbname="imdb",
        user="imdb",password="imdb")
                conn.close();
                break
            except psycopg2.OperationalError as ex:
                print("Connection failed: {0}".format(ex))
                print("""If this is the first time you are starting the
                        container, then pg_restore is probably taking its time.
                        Be patient. Will keep checking for psql to be alive.
                        Can take upto few minutes.
                        """)
                time.sleep(30)

        self.container_name = container_name

    def reset(self):
        '''
        '''
        # print("going to test cache clear thing")
        # cmd = "sh /home/pari/query-optimizer/test.sh"
        # p = sp.Popen(cmd, shell=True)
        # p.wait()

        self._send("reset")
        query = self._send("curQuery")
        if self.reward_normalization == "min_max":
            if query in self.reward_mapper:
                self._min_reward = self.reward_mapper[query][0]
                self._max_reward = self.reward_mapper[query][1]
            else:
                # FIXME: dumb hack so self.step does right thing when executing
                # random episode.
                self._min_reward = None
                self._max_reward = None
                self._min_reward, self._max_reward = self._run_random_episode()
                self.reward_mapper[query] = (self._min_reward, self._max_reward)
                # FIXME: dumb hack
                return self.reset()

        self.current_query = query
        self._observe()
        if config.qopt_viz:
            self.orig_graph = self.graph

        # clear cache if needed
        return self.graph

    def step(self, action):
        '''
        @action: edge as represented in networkX e.g., (v1,v2)
        '''
        # print("step!")
        # print(self.graph.nodes())
        # print(self.graph.edges())
        # print(action)
        # pdb.set_trace()
        # also, consider the reverse action (v1,v2) or (v2,v1) should mean the
        # same
        rev_action = (action[1], action[0])
        assert self.action_space.contains(action) or \
                    self.action_space.contains(rev_action)
        assert action in self._edge_pos_map or rev_action \
                in self._edge_pos_map
        if rev_action in self._edge_pos_map:
            action = rev_action

        action_index = self._edge_pos_map[action]

        # will make multiple zeromq calls to specify each part of the step
        # operation.
        self._send("step")
        self._send(str(action_index))
        # at this point, the calcite server would have specified the next edge
        # to be chosen in the query graph. Then, it will continue executing,
        # and block when the reward has been set

        self._observe()
        reward = float(self._send("getReward"))
        reward = self._normalize_reward(reward)
        done = int(self._send("isDone"))
        info = None
        if done:
            info = self._send("getQueryInfo")
            info = json.loads(info)
            # output episode based plots / viz
            if config.qopt_viz:
                # FIXME: this reasoning should get a lot SIMPLER once we start
                # tracking min/max scores in calcite
                if not self.orig_graph is None:
                    # only do the viz stuff now
                    next_edge = len(self.orig_graph.edges())
                    # joinOrders = info["joinOrders"]["RL"]
                    joinOrders = info["joinOrders"]["LEFT_DEEP"]
                    # now start adding stuff to it
                    # TODO: describe algorithm to create joinTree + it depends
                    # on the way we represent the graph in calcite
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    from networkx.drawing.nx_agraph import write_dot,graphviz_layout

                    G = nx.Graph()
                    root_node = None
                    for step, edge in enumerate(joinOrders):
                        # chosen edge
                        e1 = edge[0]
                        e2 = edge[1]
                        if root_node is None:
                            root_node = e1
                        G.add_edge(e1, next_edge)
                        G.add_edge(e2, next_edge)
                        next_edge += 1

                        out_name = self.viz_output_dir + "test-" + \
                            str(self.viz_ep) + "-" + str(step)
                        write_dot(G, out_name + ".dot")
                        title = self.viz_title_tmp.format(query="0",
                                ep=self.viz_ep, step=str(step))

                        plt.title(title)
                        pos = graphviz_layout(G, prog='dot', root=root_node)
                        nx.draw(G, pos, with_labels=True, arrows=True)
                        plt.savefig(out_name + ".png")
                        plt.close()

                    self.viz_ep += 1
                    pdb.set_trace()

        return self.graph, reward, done, info

    def seed(self, seed):
        print("seed! not implemented yet")

    def _handle_exit_signal(self, signum, frame):
        self.clean()

    def clean(self):
        '''
        kills the java server started by subprocess, and all it's children.
        '''
        os.killpg(os.getpgid(self.java_process.pid), signal.SIGTERM)
        print("killed the java server")
        exit(0)

    def set(self, attr, val):
        '''
        TODO: explain and reuse
        '''
        if attr == "execOnDB":
            assert config.qopt_eval_runtime or config.qopt_train_runtime
            if val:
                self._send("execOnDB")
            else:
                assert not config.qopt_train_runtime
                self._send("noExecOnDB")
        else:
            assert False

    def get_optimized_plans(self, name):
        # ignore response
        self._send(b"getOptPlan")
        resp = self._send(name)
        return resp

    def get_optimized_costs(self, name):
        self._send("getJoinsCost")
        resp = float(self._send(name))
        return resp

    def get_current_query(self):
        return self.current_query

    def _observe(self):
        '''
        TODO: more details
        gets the current query graph from calcite, and updates self.graph and
        the action space accordingly.
        '''
        # get first observation
        vertexes = self._send("getQueryGraph")
        # FIXME: hack
        vertexes = vertexes.replace("null,", "")
        vertexes = eval(vertexes)
        edges = self._send("")
        edges = eval(edges)
        graph = DirectedGraph()
        # now, let's fill it up
        nodes = {}
        # TODO: adding other attributes to the featurization scheme.

        if config.qopt_only_attr_features:
            # print("vertexes")
            for v in vertexes:
                # print(v["id"], v["visibleAttributes"])
                graph.update_nodes({v["id"] : v["visibleAttributes"]})
            self._edge_pos_map = {}
            # print("edges: ")
            for i, e in enumerate(edges):
                # print(e["factors"], e["joinAttributes"])
                graph.update_edges({tuple(e["factors"]) : e["joinAttributes"]})
                self._edge_pos_map[tuple(e["factors"])] = i
        else:
            assert False, "no other featurization scheme supported right now"
        # assert self.observation_space.contains(graph)
        self.graph = graph

        # update the possible actions we have - should be at least one fewer
        # action since we just removed an edge
        self.action_space.update_graph(self.graph)

    def _setup_space(self):
        node_feature_space = spaces.PowerSet(set(range(self.attr_count)))
        edge_feature_space = spaces.PowerSet(set(range(self.attr_count)))
        self.observation_space = spaces.Graph(node_feature_space,
                edge_feature_space)
        # we will be updating the action space as the graph evolves
        self.action_space = spaces.EdgeInGraph()

    def _start_java_server(self):
        JAVA_EXEC_FORMAT = 'mvn -e exec:java -Dexec.mainClass=Main \
        -Dexec.args="-query {query} -port {port} -train {train} \
        -lopt {lopt} -exhaustive {exh} -leftDeep {ld} -python 1 \
        -verbose {verbose} -costModel {cm} -dataset {ds} \
        -execOnDB {execOnDB} -clearCache {clearCache} \
        -recomputeFixedPlanners {recompute}"'
        # FIXME: setting the java directory relative to the directory we are
        # executing it from?
        cmd = JAVA_EXEC_FORMAT.format(
                query = config.qopt_query,
                port  = str(self.port),
                train = config.qopt_train,
                lopt = config.qopt_lopt,
                exh = config.qopt_exh,
                ld = config.qopt_left_deep,
                verbose = config.qopt_verbose,
                cm = config.qopt_cost_model,
                ds = config.qopt_dataset,
                execOnDB = config.qopt_train_runtime,
                clearCache = config.qopt_clear_cache,
                recompute = config.qopt_recompute_fixed_planners)
        try:
            qopt_path = os.environ["QUERY_OPT_PATH"]
        except:
            # if it has not been set, then set it based on the base dir
            qopt_path = self.base_dir + "/query-optimizer"

        # FIXME: hardcoded cwd, shell=False.
        # Important to use preexec_fn=os.setsid, as this puts the java process
        # and all it's children into a new groupid, which can be killed in
        # clean without shutting down the current python process
        if not config.qopt_java_output:
            FNULL = open(os.devnull, 'w')
            self.java_process = sp.Popen(cmd, shell=True,
                    cwd=qopt_path, stdout=FNULL,
                    preexec_fn=os.setsid)
        else:
            self.java_process = sp.Popen(cmd, shell=True,
                    cwd=qopt_path, preexec_fn=os.setsid)
        # FIXME: prob not required
        time.sleep(2)

    def _send(self, msg):
        """
        """
        self.socket.send_string(msg)
        ret = self.socket.recv()
        ret = ret.decode("utf8")
        return ret

    def _run_random_episode(self):
        '''
        runs a random episode, and returns the minimum / and maximum reward
        seen during this episode.
        '''
        done = False
        min_reward = 10000000
        max_reward = -10000000
        self._observe()
        while not done:
            act = self.action_space.sample()
            _, reward, done, _ = self.step(act)
            if reward < min_reward:
                min_reward = reward
            if reward > max_reward:
                max_reward = reward
        return min_reward, max_reward

    def _normalize_reward(self, reward):
        '''
        '''
        # first check if it is one of the cases in which we should just return
        # the reward without any normalization
        # if self.only_final_reward:
            # return reward
        if self._min_reward is None or self._max_reward is None:
            return reward

        if self.reward_normalization == "min_max":
            # reward = (reward-self._min_reward) / \
                    # float((self._max_reward-self._min_reward))
            reward = np.interp(reward, [self._min_reward, self._max_reward],
                    [0,1])

        elif self.reward_normalization == "":
            return reward
        else:
            assert False

        return reward
