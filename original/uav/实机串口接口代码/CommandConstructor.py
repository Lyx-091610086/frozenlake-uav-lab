from queue import Queue
from struct import pack, unpack, pack_into, unpack_from
from enum import Enum

from QueueSignal import QueueSignal


class CmdType(Enum):
    MULTI_SETTINGS = 1
    SINGLE_SETTINGS = 2
    SINGLE_CONTROL = 3
    READ_SETTINGS = 4
    pass


class CommandConstructorCore:
    """
    this class impl core read functions
    """

    # last order was generate
    order_last = 0
    # write queue
    q_write: Queue = None

    def _order_count(self):
        """
        the order_count generator, it generates new order number
        """
        self.order_last = (self.order_last + 1) % 127
        return self.order_last
        pass

    def _check_sum(self, header: bytearray, params: bytearray):
        """
        impl checksum algorithm
        """
        return bytearray([sum(header + params) & 0xFF])
        pass

    def sendCommand(self, data: bytearray):
        """
        direct do the command bytearray send task
        """
        self.q_write.put((QueueSignal.CMD, data), block=True)
        pass

    def __init__(self, q_write: Queue):
        self.q_write = q_write
        pass

    def join_cmd(self, type: CmdType, params: bytearray):
        """
        The join_cmd function concatenates the header, params and checksum
        to a cmd bytearray to let it become a valid serial cmd package.
        """

        header = bytearray(b'\xBB\x1D')

        data_body = bytearray(29)
        data_body[0] = 0xF3

        if len(params) > 13:
            raise ValueError(f"params length {len(params)} exceeds maximum 13 bytes")

        data_body[1:1 + len(params)] = params

        data_body[14:14 + len(params)] = params

        data_body[27] = 0x64  # volume = 100
        data_body[28] = 0x00  # reserved = 0

        checksum = self._check_sum(header, data_body)

        return header + data_body + checksum

    pass


class CommandConstructor(CommandConstructorCore):
    """
    this class extends CommandConstructorCore,
    it impl all functions that direct construct command .
    """

    _CMD_TAKEOFF_CODE = 0x00
    _CMD_LAND_CODE = 0xFE

    _CMD_MOVE = 0x05
    _CMD_FLIP = 0x0C
    _CMD_ARRIVE = 0x09
    _CMD_ROTATE = 0x0A
    _CMD_SPEED = 0x02
    _CMD_HIGH = 0x0B
    _CMD_AIRPLANE_MODE = 0x03
    _CMD_HOVER = 0xFE
    _CMD_LED = 0x0D

    def __init__(self, q_write: Queue):
        super().__init__(q_write)
        pass


    def _base_params(self, cmd_code: int) -> bytearray:

        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)              # id
        pack_into('<B', params, 1, cmd_code & 0xFF)   # cmd
        pack_into('<B', params, 2, self._order_count())
        return params

    # ------------ LED 相关 ------------

    def led(self, mode, r, g, b):

        if mode < 0 or mode > 2:
            raise ValueError("mode illegal", mode)

        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))

        params = self._base_params(self._CMD_LED)

        pack_into('<B', params, 3, mode)
        pack_into('<B', params, 4, r)
        pack_into('<B', params, 5, g)
        pack_into('<B', params, 6, b)

        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("led", cmd.hex(' '))
        self.sendCommand(cmd)


    def takeoff(self, high: int, ):

        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 0)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<h', params, 3, high)
        pack_into('<B', params, 5, 50)
        pack_into('<B', params, 6, 0)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("take_off", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def land(self, ):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 254)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 0)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("land", cmd.hex(' '))
        self.sendCommand(cmd)
        pass


    def move(self, direction: int, distance: int):

        if direction < 0 or direction > 10:
            raise ValueError("direction illegal", direction)
        params = self._base_params(self._CMD_MOVE)
        # direction: 0~10
        pack_into('<B', params, 3, direction & 0xFF)
        # distance: 以 cm 为单位的有符号 16 位整数
        pack_into('<h', params, 4, int(distance))

        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("move", cmd.hex(' '))
        self.sendCommand(cmd)

    def up(self, distance: int):
        # 约定 direction: 0=up
        self.move(0, distance)

    def down(self, distance: int):
        # 约定 direction: 1=down
        self.move(1, distance)

    def forward(self, distance: int):
        # 约定 direction: 2=forward
        self.move(2, distance)

    def back(self, distance: int):
        # 约定 direction: 3=back
        self.move(3, distance)

    def left(self, distance: int):
        # 约定 direction: 4=left
        self.move(4, distance)

    def right(self, distance: int):
        # 约定 direction: 5=right
        self.move(5, distance)

    # ------------ 翻滚相关 ------------

    def flip(self, direction: int, circle: int):

        if direction < 1 or direction > 4:
            raise ValueError("direction illegal", direction)
        if circle != 1 and circle != 2:
            raise ValueError("circle illegal", circle)

        params = self._base_params(self._CMD_FLIP)
        # direction: 1=front,2=back,3=left,4=right
        pack_into('<B', params, 3, direction & 0xFF)
        # circle: 1 or 2
        pack_into('<B', params, 4, circle & 0xFF)

        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("flip", cmd.hex(' '))
        self.sendCommand(cmd)

    def flip_forward(self, circle: int):
        self.flip(1, circle)

    def flip_back(self, circle: int):
        self.flip(2, circle)

    def flip_left(self, circle: int):
        self.flip(3, circle)

    def flip_right(self, circle: int):
        self.flip(4, circle)


    def arrive(self, x: int, y: int, z: int):

        params = self._base_params(self._CMD_ARRIVE)
        pack_into('<h', params, 3, int(x))
        pack_into('<h', params, 5, int(y))
        pack_into('<h', params, 7, int(z))

        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("arrive", cmd.hex(' '))
        self.sendCommand(cmd)

    def rotate(self, degree: int):

        params = self._base_params(self._CMD_ROTATE)
        pack_into('<h', params, 3, int(degree))

        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("rotate", cmd.hex(' '))
        self.sendCommand(cmd)

    def speed(self, speed: int):

        if speed < 0 or speed > 200:
            raise ValueError("speed illegal", speed)
        params = self._base_params(self._CMD_SPEED)
        pack_into('<h', params, 3, int(speed))

        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("speed", cmd.hex(' '))
        self.sendCommand(cmd)

    def high(self, high: int):

        params = self._base_params(self._CMD_HIGH)
        pack_into('<h', params, 3, int(high))

        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("high", cmd.hex(' '))
        self.sendCommand(cmd)

    def airplane_mode(self, mode: int):

        if mode < 1 or mode > 4:
            raise ValueError("mode illegal", mode)
        params = self._base_params(self._CMD_AIRPLANE_MODE)
        pack_into('<B', params, 3, mode & 0xFF)

        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("airplane_mode", cmd.hex(' '))
        self.sendCommand(cmd)

    def hovering(self, ):

        params = self._base_params(self._CMD_HOVER)

        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("hovering", cmd.hex(' '))
        self.sendCommand(cmd)



    def read_multi_setting(self):
        return self.read_setting(0x02)

    def read_single_setting(self):
        return self.read_setting(0x04)

    def read_hardware_setting(self):
        return self.read_setting(0xA0)

    def read_setting(self, mode: int):
        params = bytearray(1)
        pack_into("!B", params, 0, mode)
        cmd = self.join_cmd(CmdType.READ_SETTINGS, params)
        print("cmd", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    pass
