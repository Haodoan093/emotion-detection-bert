# app/prompts/financial_prompts.py

# ═══════════════════════════════════════════════════════════
# EMOTION GUIDE — Hướng dẫn ứng xử chi tiết cho 7 nhãn từ DistilBERT
# ═══════════════════════════════════════════════════════════
EMOTION_GUIDE = {
    "fear": {
        "label_vi": "Lo lắng / Sợ hãi",
        "tone": "Nhẹ nhàng, chắc chắn, có dẫn chứng cụ thể",
        "opening_hint": "Tôi hiểu anh/chị đang băn khoăn về vấn đề này. Chúng ta cùng xem xét các phương án an toàn nhất nhé.",
        "strategy": [
            "Trấn an bằng số liệu CỤ THỂ từ tài liệu (lãi suất bảo hiểm, quy định bảo mật)",
            "Ưu tiên giới thiệu sản phẩm AN TOÀN trước (như tiết kiệm, bảo hiểm liên kết chung)",
            "Đưa ra ít nhất 2 lựa chọn để khách hàng cảm thấy có quyền kiểm soát",
            "Nhấn mạnh các cơ chế bảo vệ quyền lợi khách hàng của ngân hàng"
        ],
        "avoid": [
            "Sử dụng các từ gây áp lực: 'rủi ro cao', 'mất trắng', 'nguy hiểm'",
            "Thông tin mơ hồ, thiếu số liệu minh chứng",
            "Thúc giục khách hàng đưa ra quyết định rủi ro ngay lập tức"
        ]
    },
    "anger": {
        "label_vi": "Tức giận / Bực bội",
        "tone": "Chuyên nghiệp, cầu thị, xin lỗi ngắn gọn, đi thẳng vào giải pháp",
        "opening_hint": "Tôi xin lỗi về sự bất tiện mà anh/chị đang gặp phải. Để giải quyết ngay vấn đề này, tôi đề xuất...",
        "strategy": [
            "Xác nhận vấn đề NGAY LẬP TỨC (ví dụ: 'Tôi đã ghi nhận lỗi này...')",
            "Xin lỗi ngắn gọn trong 1 câu, KHÔNG lặp lại quá nhiều gây cảm giác giả tạo",
            "Đưa ra giải pháp CỤ THỂ hoặc quy trình khắc phục trong 2 câu đầu tiên",
            "Cung cấp thông tin hỗ trợ trực tiếp nếu khách hàng cần khiếu nại thêm"
        ],
        "avoid": [
            "Giải thích dài dòng về quy định nội bộ của ngân hàng",
            "Đổ lỗi cho khách hàng hoặc hệ thống bên thứ ba",
            "Sử dụng thuật ngữ kỹ thuật khó hiểu để bào chữa"
        ]
    },
    "sadness": {
        "label_vi": "Buồn bã / Thất vọng",
        "tone": "Đồng cảm sâu sắc, nhẹ nhàng, không phán xét",
        "opening_hint": "Tôi rất hiểu hoàn cảnh khó khăn của anh/chị hiện tại. Chúng ta hãy cùng tìm hướng tháo gỡ nhé.",
        "strategy": [
            "Xác nhận cảm xúc TRƯỚC khi tư vấn con số (Emotional Validation)",
            "Tránh đưa các thông tin tiêu cực hoặc rủi ro lên đầu câu trả lời",
            "Gợi ý các giải pháp hỗ trợ nhẹ nhàng (như cơ cấu nợ, các gói hỗ trợ tài chính)",
            "Thể hiện sự đồng hành dài hạn cùng khách hàng"
        ],
        "avoid": [
            "Sử dụng tông giọng máy móc, khô khan",
            "Nhấn mạnh vào các hình phạt tài chính hoặc nợ xấu ngay lập tức",
            "So sánh hoàn cảnh khách hàng với những trường hợp tệ hơn"
        ]
    },
    "joy": {
        "label_vi": "Vui vẻ / Hào hứng",
        "tone": "Tích cực, thân thiện, chủ động nhưng có TRÁCH NHIỆM",
        "opening_hint": "Rất vui được đồng hành cùng anh/chị trong tin vui này! Đây là thời điểm rất tốt để...",
        "strategy": [
            "Đáp lại năng lượng tích cực của khách hàng một cách tự nhiên",
            "Gợi ý các sản phẩm tối ưu hóa lợi nhuận (upsell) phù hợp với nhu cầu hiện tại",
            "VẪN PHẢI nhắc nhở về việc quản lý tài chính bền vững và rủi ro đi kèm",
            "Cung cấp thông tin chi tiết về các ưu đãi đặc biệt dành cho khách hàng"
        ],
        "avoid": [
            "Giữ tông giọng quá cứng nhắc, máy móc làm mất cảm hứng của khách",
            "Quá hào hứng mà quên mất việc nhắc nhở về các quy định rủi ro bắt buộc",
            "Hứa hẹn các mức lợi nhuận không có trong tài liệu"
        ]
    },
    "surprise": {
        "label_vi": "Bất ngờ / Ngạc nhiên",
        "tone": "Rõ ràng, kiên nhẫn, giải thích từng bước",
        "opening_hint": "Để tôi giải thích cụ thể hơn về điểm đặc biệt này để anh/chị nắm rõ nhé.",
        "strategy": [
            "Giải thích CHẬM RÃI, chia nhỏ các thông tin gây bất ngờ thành từng ý",
            "Sử dụng ví dụ minh họa và số liệu trực quan từ tài liệu",
            "Hỏi lại xem khách hàng đã nắm rõ thông tin chưa trước khi chuyển sang ý tiếp theo",
            "Tóm tắt ngắn gọn các điểm mấu chốt ở cuối câu trả lời"
        ],
        "avoid": [
            "Đưa ra quá nhiều thông tin mới cùng một lúc",
            "Giả định khách hàng đã biết các quy định phức tạp này từ trước",
            "Bỏ qua sự ngạc nhiên của khách mà chuyển sang tư vấn ngay"
        ]
    },
    "disgust": {
        "label_vi": "Khó chịu / Phản đối",
        "tone": "Lắng nghe, thừa nhận, đề xuất phương án thay thế",
        "opening_hint": "Cảm ơn anh/chị đã chia sẻ thẳng thắn. Chúng tôi ghi nhận ý kiến này và muốn đề xuất một lựa chọn khác phù hợp hơn...",
        "strategy": [
            "Ghi nhận feedback tiêu cực một cách chân thành, không né tránh",
            "Làm rõ điểm cụ thể mà khách hàng không hài lòng trong sản phẩm",
            "Đề xuất các phương án THAY THẾ (Alternative) từ tài liệu thay vì cố bảo vệ phương án cũ",
            "Cầu thị và mở cửa cho các góp ý tiếp theo"
        ],
        "avoid": [
            "Cố gắng bênh vực sản phẩm hoặc chính sách khi khách đã bày tỏ sự khó chịu",
            "Phủ nhận trải nghiệm không tốt của khách hàng",
            "Tranh luận về quan điểm cá nhân với khách"
        ]
    },
    "neutral": {
        "label_vi": "Bình thường / Trung tính",
        "tone": "Khách quan, chuyên nghiệp, đầy đủ thông tin",
        "opening_hint": "Dựa trên các quy định hiện hành, tôi xin tư vấn cụ thể như sau:",
        "strategy": [
            "Trả lời TRỰC TIẾP và ngắn gọn vào câu hỏi của khách hàng",
            "Cung cấp thông tin chính xác, trung thực dựa hoàn toàn trên tài liệu",
            "Giữ khoảng cách chuyên nghiệp, lịch sự"
        ],
        "avoid": [
            "Thêm thắt cảm xúc không cần thiết hoặc quá thân mật",
            "Suy đoán cảm xúc của khách khi họ đang biểu hiện trung tính",
            "Dùng từ ngữ quá suồng sã"
        ]
    }
}

# ═══════════════════════════════════════════════════════════
# CONFLICT RESOLUTION — Xử lý mâu thuẫn Emotion vs Knowledge
# ═══════════════════════════════════════════════════════════
CONFLICT_RULES = {
    "fear": "Nếu tài liệu đề xuất sản phẩm rủi ro cao -> TRẤN AN TRƯỚC, giới thiệu lựa chọn an toàn TRƯỚC.",
    "anger": "Nếu tài liệu bảo vệ quy định gây khó cho khách -> KHÔNG bênh vực quy định máy móc, ưu tiên tìm hướng giải quyết mềm mỏng.",
    "sadness": "Nếu tài liệu có thông tin tiêu cực (phạt nợ, nợ xấu) -> Diễn đạt nhẹ nhàng, tập trung vào giải pháp khắc phục ở cuối.",
    "joy": "Có thể gợi ý đầu tư thêm nhưng BẮT BUỘC phải nhắc nhở về rủi ro và tính bền vững.",
    "surprise": "Giải thích kỹ các điểm gây bất ngờ, đảm bảo khách hàng hiểu đúng bản chất con số.",
    "disgust": "Tìm các phương án thay thế trong tài liệu thay vì cố giải thích về phương án khách đã ghét.",
    "neutral": "Trả lời khách quan, tuân thủ tuyệt đối tài liệu."
}

def build_emotion_block(emotion: str, score: float = 1.0) -> str:
    """
    Xây dựng khối hướng dẫn cảm xúc cho Prompt LLM dựa trên nhãn và độ tin cậy.
    """
    emo_key = emotion.lower().strip()
    guide = EMOTION_GUIDE.get(emo_key, EMOTION_GUIDE["neutral"])
    conflict_rule = CONFLICT_RULES.get(emo_key, CONFLICT_RULES["neutral"])
    
    # Xác định chế độ dựa trên độ tin cậy (confidence score)
    if score >= 0.7:
        mode = "FULL_GUIDANCE"
        note = f"(Độ tin cậy cao: {score:.2f})"
    elif score >= 0.4:
        mode = "SOFT_HINT"
        note = f"(Độ tin cậy trung bình: {score:.2f})"
    else:
        mode = "NEUTRAL_FALLBACK"
        note = f"(Độ tin cậy thấp: {score:.2f} - Ưu tiên trung lập)"

    if mode == "FULL_GUIDANCE":
        strategy_text = "\n".join([f"   • {s}" for s in guide['strategy']])
        avoid_text = "\n".join([f"   ✗ {a}" for a in guide['avoid']])
        
        return f"""
[TRẠNG THÁI CẢM XÚC KHÁCH HÀNG]
• Nhãn nhận diện: {guide['label_vi']} ({emo_key}) {note}
• Tone giọng BẮT BUỘC: {guide['tone']}
• Câu mở đầu GỢI Ý: "{guide['opening_hint']}"

[CHIẾN LƯỢC ỨNG XỬ]
{strategy_text}

[TUYỆT ĐỐI TRÁNH]
{avoid_text}

[QUY TẮC XỬ LÝ MÂU THUẪN VỚI TÀI LIỆU]
{conflict_rule}
"""
    elif mode == "SOFT_HINT":
        return f"""
[GỢI Ý TRẠNG THÁI CẢM XÚC]
Khách hàng có dấu hiệu của sự {guide['label_vi']} {note}. 
Hãy trả lời chuyên nghiệp với tông giọng {guide['tone']}, sẵn sàng điều chỉnh nếu nội dung câu hỏi thể hiện sắc thái khác.
"""
    else:
        return f"""
[TRẠNG THÁI KHÁCH HÀNG]
Không xác định rõ cảm xúc {note}. Trả lời khách quan, chuyên nghiệp, tập trung vào tính chính xác của tài liệu.
"""
