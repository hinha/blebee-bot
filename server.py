import grpc
from concurrent import futures

import src.pb.chat_pb2_grpc as pb2_grpc
import src.pb.chat_pb2 as pb2

from src.bot import TrainBot


class ChatService(pb2_grpc.chatBot):

    def retrieveMessage(self, request, context):
        bot = TrainBot()
        model = bot.training()
        clas = bot.classify(request.message, model)
        result = {'message': clas[0], 'label': clas[1], 'accuracy': clas[3]}

        return pb2.chatResponse(**result)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_chatBotServicer_to_server(ChatService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()




if __name__ == '__main__':
    serve()
