import py4j.GatewayServer;

public class SimpleNLGWrapper {

  public int addition(int first, int second) {
    return first + second;
  }

  public static void main(String[] args) {
    SimpleNLGWrapper app = new SimpleNLGWrapper();
    // app is now the gateway.entry_point
    GatewayServer server = new GatewayServer(app);
    server.start();
  }
}