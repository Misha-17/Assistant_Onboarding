"""Composite web facade adding an explicitly requested post-answer audit."""
from http import HTTPStatus
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit
import webbrowser

from sisu_reader import web as core
from .service import AuditService, AuditError

AUDIT_ROUTE=re.compile(r'/api/jobs/(?P<job>job_[A-Za-z0-9_-]{20,160})/audit')


class Application(core._Application):
    def __init__(self,config,*,audit_client_factory=None,audit_runtime_identity_provider=None,audit_budget_s=20.0,audit_startup_budget_s=60.0):
        super().__init__(config)
        self.assets_dir=Path(__file__).resolve().parent/'web_assets'
        self.citation_audits=AuditService(self,client_factory=audit_client_factory,runtime_identity_provider=audit_runtime_identity_provider,budget_s=audit_budget_s,startup_budget_s=audit_startup_budget_s)

    def close(self):
        service=getattr(self,'citation_audits',None)
        if service is not None:service.close()
        super().close()


class Handler(core._Handler):
    def _audit(self,*,start):
        match=AUDIT_ROUTE.fullmatch(urlsplit(self.path).path)
        if match is None:return False
        if not self._guard():return True
        browser=self._browser()
        if browser is None:return True
        if start:
            body=self._read_json()
            if body is None:return True
            if body:
                self._error(HTTPStatus.BAD_REQUEST,'audit_body','The audit uses the saved answer; send an empty JSON object.')
                return True
        try:
            method=self.application.citation_audits.start if start else self.application.citation_audits.get
            value=method(match['job'],browser)
            self._send_json(HTTPStatus.ACCEPTED if start and value['status'] in {'queued','running'} else HTTPStatus.OK,{'audit':value})
        except AuditError as exc:self._error(exc.status,exc.code,exc.message)
        return True

    def do_GET(self):
        if not self._audit(start=False):super().do_GET()

    def do_POST(self):
        if not self._audit(start=True):super().do_POST()


def run_web(config,*,open_browser=True):
    if not core._loopback_host(config.web_host):raise ValueError('SISU Reader UI must bind to loopback')
    application=Application(config)
    handler=type('SisuAuditHandler',(Handler,),{'application':application})
    try:server=core._Server((config.web_host,config.web_port),handler)
    except Exception:application.close();raise
    host=f'[{config.web_host}]' if ':' in config.web_host else config.web_host
    url=f'http://{host}:{config.web_port}/'
    print(f'SISU Reader UI with optional citation audit: {url}')
    if open_browser:threading.Timer(.25,lambda:webbrowser.open(url)).start()
    try:server.serve_forever(poll_interval=.25)
    finally:server.server_close();application.close()
