import serial
from queue import Queue, Empty
from time import sleep
from threading import Thread

from ReadDataParser import ReadDataParser, Fh0cBase
from CommandConstructor import CommandConstructor
from QueueSignal import QueueSignal


class ThreadLocal:
    latest_cmd: bytearray = None
    q: Queue = None
    s: serial.Serial = None
    t: Thread = None
    exit_queue: Queue = Queue()

    rdp: ReadDataParser = None

    def __init__(self):
        pass


def task_write(thread_local: ThreadLocal):
    print("[task_write] start")
    while True:
        sleep(0.1)

        try:
            if thread_local.exit_queue.get(block=False) is QueueSignal.SHUTDOWN:
                break
        except Empty:
            pass

        try:
            d = thread_local.q.get(block=False)
            if isinstance(d, tuple):
                print("[task_write] Tuple:", (d[0], d[1].hex(" ")))
                if d[0] is QueueSignal.CMD and len(d[1]) > 0:
                    thread_local.latest_cmd = d[1]
        except Empty:
            pass

        if thread_local.latest_cmd is not None and len(thread_local.latest_cmd) > 0:
            thread_local.s.write(thread_local.latest_cmd)

    print("[task_write] done.")


def task_read(thread_local: ThreadLocal):
    print("[task_read] start")
    while True:
        sleep(0.1)

        try:
            if thread_local.exit_queue.get(block=False) is QueueSignal.SHUTDOWN:
                break
        except Empty:
            pass

        d = thread_local.s.read(65535)
        if d:
            thread_local.rdp.push(d)

    print("[task_read] done.")


class SerialThreadCore:
    s: serial.Serial = None
    port: str = None
    thread_local_write: ThreadLocal = None
    thread_local_read: ThreadLocal = None

    def __init__(self, port: str):
        self.port = port
        self.q_write: Queue = Queue()
        self.q_read: Queue = Queue()

        print(f"[SerialThreadCore] open port: {port}")
        self.s = serial.Serial(port, baudrate=500000, timeout=0.01)


        self.thread_local_write = ThreadLocal()
        self.thread_local_write.q = self.q_write
        self.thread_local_write.s = self.s
        self.thread_local_write.t = Thread(
            target=task_write,
            args=(self.thread_local_write,),
            daemon=True
        )


        self.thread_local_read = ThreadLocal()
        self.thread_local_read.q = self.q_read
        self.thread_local_read.s = self.s
        self.thread_local_read.rdp = ReadDataParser(self.thread_local_read.q)
        self.thread_local_read.t = Thread(
            target=task_read,
            args=(self.thread_local_read,),
            daemon=True
        )

        self.thread_local_write.t.start()
        self.thread_local_read.t.start()


    def shutdown(self):
        print("[SerialThreadCore] shutdown...")

        self.thread_local_write.exit_queue.put(QueueSignal.SHUTDOWN)
        self.thread_local_read.exit_queue.put(QueueSignal.SHUTDOWN)

        self.thread_local_write.t.join()
        self.thread_local_read.t.join()

        self.s.close()
        print("[SerialThreadCore] shutdown done.")


    def fh0c_base(self) -> Fh0cBase:
        return self.thread_local_read.rdp.get_fh0c_base()


    def monitor_link(self, duration: float = 3.0, interval: float = 0.1, tag: str = ""):
        steps = int(duration / interval)
        if steps <= 0:
            steps = 1

        print(f"[monitor_link] [{tag}] start {duration:.1f}s, interval={interval:.2f}s")

        for _ in range(steps):
            base = self.fh0c_base()
            if base:
                print(
                    "[monitor_link]"
                    f" id={base.id}"
                    f" vol={base.vol}"
                    f" ssi={base.ssi}"
                    f" high={base.high}"
                    f" flow=({base.flow_x}, {base.flow_y})"
                    f" mv0=({base.mv_x0}, {base.mv_y0})"
                    f" imu={base.imu}"
                )
            sleep(interval)

        print("[monitor_link] end")


class SerialThread(SerialThreadCore):
    ss: CommandConstructor = None

    def __init__(self, port: str):
        super().__init__(port)
        self.ss = CommandConstructor(self.thread_local_write.q)
        print("[SerialThread] CommandConstructor ready")

    def send(self):
        return self.ss



