#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class PredictionSubscriber : public rclcpp::Node {
public:
    PredictionSubscriber() : Node("prediction_subscriber") {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "emotion_prediction", 10,
            std::bind(&PredictionSubscriber::topic_callback, this, std::placeholders::_1));
    }

private:
    void topic_callback(const std_msgs::msg::String::SharedPtr msg) const {
        RCLCPP_INFO(this->get_logger(), "Received prediction: '%s'", msg->data.c_str());
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    std::cout << "Prediction subscriber started" << std::endl;
    rclcpp::spin(std::make_shared<PredictionSubscriber>());
    rclcpp::shutdown();
    return 0;
}
