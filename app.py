from tornado import web,ioloop,httpserver,gen
# from findPath import getPath
import findPath
data=findPath.data
RESULT='test'

class MainPageHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        self.render('testUI.html',data=data,result='')
    def post(self, *args, **kwargs):
        startPos = self.get_argument('startPos')
        endPos = self.get_argument('endPos')
        if startPos and endPos:
            global RESULT
            RESULT = findPath.getPath(startPos, endPos)
            self.redirect('/index')
        else:
            self.write('内容不能为空')

class ShowResultHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        self.render('index.html',result=RESULT)
    def post(self, *args, **kwargs):
        self.redirect('/')

settings = {
    'template_path':'templates',
    'static_path':'static'
}

application = web.Application([
            (r"/", MainPageHandler),
            (r"/index", ShowResultHandler),
        ], **settings)

if __name__ == '__main__':
    http_server = httpserver.HTTPServer(application)
    http_server.listen(8080)
    ioloop.IOLoop.current().start()