#!/usr/bin/env python3
"""
Module MediatorLib
This module contains helper functions for dealing with the Mediator.

This module was written long after MediatorInterface and XmlSocketServer and aims to outsource functionality
that was written interweaved with the modules mentioned and could therefore not be used on its own.
"""

import logging
import re
import socket
import struct
import json
import http.client as httpc
import xml.etree.ElementTree as ET
import ssl

from typing import Union, Optional, Tuple, Type, NamedTuple

Socket  = socket.socket
XmlEl   = ET.Element

logger = logging.getLogger("__name__")

"""
Helper classes
"""

class Auth(NamedTuple):
    """
    Authentication Tuple containing the information needed to
    get a token
    """
    user: str
    passw: str
    server: str

    def __repr__(self) -> str:
        return f"(user={self.user}, passw={'***' if self.passw is not None else '<not set>'}, server={self.server}"

class MediatorAudioFormat(NamedTuple):
    rate: int
    chunksize: int
    format: int


"""
Low level helper
Sending/receiving messages
"""

def create_connection(server: str, port: int = 4443, timeout: Optional[int] = 10) -> Socket:
    """
    Create a new connection to the Mediator and return the socket
    See https://docs.python.org/library/socket.html#socket.socket.connect for
    errors thrown.
    """
    sock = Socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((server, port))
    return sock

def recv_data(sock: Socket, peek: bool = False) -> bytes:
    """
    Receive a message in bytes from the Mediator via socket `sock`.
    """
    flags = socket.MSG_PEEK if peek else 0

    # Receiving message length
    buf = sock.recv(4, flags)
    assert len(buf) == 4, "Did not get a valid message from the Mediator"
    msg_len = socket.htonl(struct.unpack("<L", buf)[0])

    # Receive actual message
    if peek:
        msg = sock.recv(msg_len + 4, flags)[4:]
    else:
        msg = sock.recv(msg_len, flags)

    assert len(msg) == msg_len, f"Expected {msg_len} bytes message from the mediator but got {len(msg)}"
    return msg

def recv_msg(sock: Socket, peek: bool = False) -> str:
    """
    Receive a message from the Mediator and interpret it as as string
    """
    return recv_data(sock, peek).decode("utf-8")

def recv_xml(sock: Socket, peek: bool = False) -> XmlEl:
    return ET.fromstring(recv_msg(sock, peek))

def send(sock: Socket, msg: Union[str, bytes, XmlEl]) -> None:
    if isinstance(msg, XmlEl):
        msg = ET.tostring(msg)
    if isinstance(msg, str):
        msg = msg.encode("ascii", "strict")

    data: bytes = msg

    msg_len_enc = struct.pack("<L", socket.htonl(len(data)))
    sock.sendall(msg_len_enc + data)

"""
Higher level functions
"""

def solve_challenge(challenge: Union[int, str]) -> int:
    if isinstance(challenge, str):
        challenge = int(challenge)
    return (challenge * 194510094) % 1999999943


def do_auth(sock: Socket, auth: Optional[Auth], check_required: bool = True) -> bool:
    """
    If the mediator requires auth, authentication will be done, otherwise
    nothing.

    If check_required is False, no data will be read from the socket and it
    will be assumed that the mediator already waits for our answer.

    If check_required is True, it will be peeked on the socket if there is a
    message and no data will be read if no auth is required. If auth is
    required, the auth request message will be read (and removed) from the
    socket.

    If `auth` does not contain a password, getpass is used to get user input.

    Returns True on success or no authentication required. Prints an error
    message on errors and returns False.
    """
    if check_required:
        msg = recv_xml(sock, peek=True)
        if ("type", "auth") not in msg.items():
            return True
        recv_xml(sock, peek=False)

    if auth is None:
        logger.error("Mediator requires authentication but no credentials provided")
        return False

    user, passw, server = auth
    if passw is None:
        from getpass import getpass
        passw = getpass(prompt=f"Password for {user}: ")

    logger.info(f"Trying to log in via {server}...")

    auth_succ, token = get_authtoken(server, user, passw)
    if not auth_succ:
        logger.error(f'Error authenticating: "{token}"')
        return False;

    logger.info(f"Got authentication. Sending.")
    logger.debug(f"Token: {token}")

    send(sock, f'<auth token="{token}"/>')
    return True


def get_authtoken(authserver: str, username: str, password: str) -> Tuple[bool, str]:
    """
    Not for mediator, connects to a ltauthserver with username and password to return a token.
    TODO WIP and should be used by mediatorInterface in the future
    Returns:
        Tuple containing a bool and string. If true, the string contains the token, otherwise an error message
    """
    connector: Optional[Union[Type[httpc.HTTPSConnection], Type[httpc.HTTPConnection]]] = None
    if authserver.startswith("https://"):
        connector = httpc.HTTPSConnection
    elif authserver.startswith("http://"):
        connector = httpc.HTTPConnection
    else:
        raise Exception(f"'{authserver}' is not a http(s) url")
    authserver = re.sub('^https?://', '', authserver)
    # authserver, url = authserver.split('/', maxsplit=1)

    payload = json.dumps({ "name": username, "password": password })
    headers = {"Content-type": "application/json", "Host": authserver}

    if connector == httpc.HTTPSConnection:
        context = ssl.SSLContext() #ssl._create_unverified_context()
        context.load_default_certs()
        connection = connector(authserver, context=context)
    else:
        connection = connector(authserver)

    connection.request("POST" , '/auth/mediator/', payload, headers)
    with connection.getresponse() as resp:
        if resp.status // 100 != 2:
            return (False, f"Authserver '{authserver}' returned code {resp.status}: '{resp.reason}', '{resp.read().decode()}'")

        body = json.loads(resp.read().decode())
        assert "token" in body
        return (True, body["token"])


def get_worker_list(server: str, port: int,
        auth: Optional[Auth]) -> Optional[str]:
    """
    Opens a new connection to the mediator and returns the json string
    containing all workers according to protocol.
    """
    assert server is not None, "No Mediator servername/address given"

    sock = create_connection(server, port)
    if not do_auth(sock, auth, check_required=True):
        return None

    resp = recv_xml(sock)

    # Do challenge
    assert resp.tag == 'status' and ("type", "connect") in resp.items(), (
            f"Expected 'status' tag with 'connect' type in response from Mediator. Got: '{resp}'"
    )
    challenge = resp.attrib.get('description')
    assert challenge is not None, (
            "Expected 'description' attribute in response from Mediator."
    )
    solution = solve_challenge(challenge)

    # Request connected workers
    send(sock, f"<availablequeues sessionid=\"{solution}\"/>")
    workers = recv_msg(sock)
    sock.close()

    return workers
