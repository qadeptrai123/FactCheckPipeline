# Tài liệu cấu trúc output cho bước refine input

Tài liệu này mô tả JSON output của bước `refined/refine_input.py`. Mục tiêu của output là biến một claim tiếng Việt, có thể kèm ảnh, thành biểu diễn có cấu trúc để phục vụ RAG, truy hồi đa phương thức và kiểm chứng từng mệnh đề.

## Nguyên tắc chung

- Tất cả nội dung sinh ra nên viết bằng tiếng Việt.
- Không kết luận claim đúng hay sai ở bước refine.
- Không thêm thông tin không có trong claim hoặc không nhìn thấy trong ảnh.
- Nếu có ảnh, chỉ mô tả bằng chứng nhìn thấy trực tiếp.
- Nếu không có ảnh, `visual_observations` phải là mảng rỗng và `alignment.label` nên là `not_enough_visual_info`.
- Các trường dạng mảng dùng `[]` khi không có thông tin.
- Các query sinh ra chỉ phục vụ tìm bằng chứng, không nhắc đến vector database, embedding model hay backend retrieval.

## Cấu trúc tổng thể

```json
{
  "original_claim": "...",
  "normalized_claim": "...",
  "primary_retrieval_query": "...",
  "image_provided": true,
  "language": "vi",
  "claim_atoms": [],
  "visual_observations": [],
  "alignment": {},
  "key_entities": {},
  "search_queries": {},
  "retrieval_focus": {},
  "constraints": {},
  "context_summary": "...",
  "ambiguity_notes": [],
  "verification_targets": []
}
```

## Mô tả từng trường

### `original_claim`

Claim gốc nhận từ input. Trường này giữ lại nội dung ban đầu để đối chiếu, debug và đánh giá xem refine có làm lệch nghĩa hay không.

Ví dụ:

```json
"original_claim": "Công an tỉnh Thái Bình đã khởi tố 10 đối tượng trong đường dây lừa đảo hỗ trợ vay vốn online."
```

### `normalized_claim`

Phiên bản claim đã chuẩn hóa nhưng không đổi nghĩa. Có thể sửa lỗi gõ, chuẩn hóa dấu câu, làm rõ câu văn, nhưng không được thêm kết luận mới.

Dùng cho:

- Baseline query cho dense retrieval.
- Hiển thị cho người dùng.
- So sánh với claim gốc khi debug.

### `primary_retrieval_query`

Truy vấn chính tốt nhất để tìm bằng chứng cho toàn bộ input. Đây là trường nên dùng đầu tiên khi bạn cần một query duy nhất cho RAG.

Khác với `normalized_claim`: trường này được tối ưu cho truy hồi, nên có thể sắp xếp lại entity, sự kiện, thời gian, địa điểm để dễ tìm kiếm hơn.

Ví dụ:

```json
"primary_retrieval_query": "Công an tỉnh Thái Bình khởi tố 10 đối tượng lừa đảo hỗ trợ vay vốn online"
```

### `image_provided`

Cho biết input có ảnh hợp lệ được đưa vào model hay không.

- `true`: có ảnh và ảnh được đọc thành công.
- `false`: không có ảnh hoặc đường dẫn ảnh không được resolve.

Lưu ý: nếu đường dẫn ảnh sai, script hiện tại xem như không có ảnh. Khi đánh giá dataset, nên kiểm tra thêm cột lỗi nếu bạn bổ sung sau này.

### `language`

Ngôn ngữ của output. Hiện tại schema chỉ chấp nhận:

```json
"language": "vi"
```

### `claim_atoms`

Danh sách các mệnh đề nguyên tử cần kiểm chứng. Mỗi atom nên là một ý có thể tìm bằng chứng riêng.

Mỗi phần tử có cấu trúc:

```json
{
  "id": "c1",
  "text": "Công an tỉnh Thái Bình đã khởi tố 10 đối tượng.",
  "check_type": "event",
  "priority": "high",
  "retrieval_queries": [
    "Công an Thái Bình khởi tố 10 đối tượng",
    "Thái Bình khởi tố 10 bị can lừa đảo vay vốn online"
  ]
}
```

#### `claim_atoms[].id`

Mã định danh của atom. Nên dùng dạng `c1`, `c2`, `c3`.

#### `claim_atoms[].text`

Nội dung mệnh đề có thể kiểm chứng. Nên ngắn gọn, rõ chủ thể, hành động, đối tượng, thời gian hoặc địa điểm nếu có.

#### `claim_atoms[].check_type`

Loại thông tin cần kiểm chứng:

- `entity`: danh tính người, tổ chức, đối tượng.
- `event`: sự kiện, hành động, vụ việc.
- `time`: thời gian, ngày tháng, mốc lịch sử.
- `location`: địa điểm.
- `number`: số liệu, số lượng, tỷ lệ, tiền.
- `quote`: phát ngôn, câu trích dẫn.
- `relation`: mối quan hệ giữa các entity.
- `other`: trường hợp khác.

#### `claim_atoms[].priority`

Mức ưu tiên kiểm chứng:

- `high`: ý chính, ảnh hưởng lớn đến kết luận.
- `medium`: ý quan trọng nhưng không phải trọng tâm.
- `low`: chi tiết phụ, nên kiểm chứng nếu có bằng chứng.

#### `claim_atoms[].retrieval_queries`

Danh sách query riêng cho atom đó. Trường này rất quan trọng cho RAG testing vì nó cho phép đánh giá retrieval theo từng mệnh đề thay vì chỉ theo claim dài.

Nên tạo 1-3 query cho mỗi atom, bao gồm biến thể semantic và từ khóa quan trọng.

### `visual_observations`

Danh sách quan sát trực tiếp từ ảnh. Chỉ dùng khi có ảnh.

Mỗi phần tử có cấu trúc:

```json
{
  "id": "v1",
  "text": "Ảnh cho thấy một nhóm người đứng trong phòng họp.",
  "visible_evidence": ["nhóm người", "phòng họp", "bàn đại biểu"],
  "confidence": "high"
}
```

#### `visual_observations[].id`

Mã định danh của quan sát, nên dùng `v1`, `v2`, `v3`.

#### `visual_observations[].text`

Mô tả ngắn gọn điều nhìn thấy trong ảnh. Không suy diễn danh tính, ý định, nguyên nhân, thời điểm hoặc sự kiện ngoài khung hình.

#### `visual_observations[].visible_evidence`

Danh sách chi tiết thị giác làm cơ sở cho quan sát, ví dụ: văn bản trên biển, logo, đồng phục, vật thể, bối cảnh, số lượng ước tính.

#### `visual_observations[].confidence`

Độ tin cậy của quan sát:

- `high`: thấy rõ.
- `medium`: khá rõ nhưng có thể bị che, mờ, cắt khung.
- `low`: khó nhìn, không chắc chắn.

### `alignment`

Đánh giá mối quan hệ giữa claim và bằng chứng nhìn thấy trong ảnh. Đây không phải là kết luận fact-check cuối cùng.

Cấu trúc:

```json
{
  "label": "partial_match",
  "text": "Ảnh có một số chi tiết phù hợp với claim, nhưng không đủ thông tin để xác nhận toàn bộ nội dung."
}
```

#### `alignment.label`

Giá trị hợp lệ:

- `match`: ảnh có vẻ phù hợp với claim ở các chi tiết nhìn thấy.
- `partial_match`: ảnh chỉ hỗ trợ một phần claim.
- `mismatch`: chi tiết nhìn thấy mâu thuẫn với claim.
- `not_enough_visual_info`: ảnh không đủ thông tin, hoặc không có ảnh.

#### `alignment.text`

Giải thích ngắn gọn vì sao gán label đó, dựa trên chi tiết nhìn thấy.

### `key_entities`

Danh sách entity và giá trị quan trọng được trích xuất từ claim và ảnh.

Cấu trúc:

```json
{
  "people": [],
  "organizations": ["Công an tỉnh Thái Bình"],
  "locations": ["Thái Bình"],
  "dates": [],
  "numbers": ["10"],
  "other": ["vay vốn online", "lừa đảo"]
}
```

Dùng cho:

- Query expansion.
- Keyword retrieval.
- Lọc theo metadata nếu corpus có metadata.
- Debug lỗi retrieval do thiếu entity.

### `search_queries`

Tập query tổng quát cho các chiến lược retrieval khác nhau.

Cấu trúc:

```json
{
  "semantic": ["Công an Thái Bình khởi tố các đối tượng lừa đảo vay vốn online"],
  "keywords": ["Công an Thái Bình", "khởi tố", "10 đối tượng", "vay vốn online"],
  "visual": ["ảnh họp báo công an với nhóm người và tang vật điện thoại máy tính"]
}
```

#### `search_queries.semantic`

Query câu đầy đủ, tự nhiên, phù hợp dense retrieval.

#### `search_queries.keywords`

Từ khóa ngắn, entity, số liệu, cụm từ đặc trưng, phù hợp sparse retrieval, BM25 hoặc hybrid retrieval.

#### `search_queries.visual`

Mô tả thị giác cần tìm, phù hợp truy hồi ảnh, caption retrieval hoặc cross-modal retrieval. Nếu không có ảnh và claim không cần bằng chứng hình ảnh, có thể để `[]`.

### `retrieval_focus`

Chỉ ra modality nào nên được dùng khi truy hồi.

Cấu trúc:

```json
{
  "text": true,
  "image": true,
  "cross_modal": true
}
```

#### `retrieval_focus.text`

`true` nếu cần tìm bằng chứng văn bản, bài báo, thông báo, tài liệu.

#### `retrieval_focus.image`

`true` nếu ảnh là bằng chứng quan trọng hoặc claim liên quan đến nội dung hình ảnh.

#### `retrieval_focus.cross_modal`

`true` nếu cần dùng text để tìm ảnh, hoặc dùng mô tả ảnh để tìm văn bản liên quan.

### `constraints`

Ràng buộc cần tôn trọng khi retrieval. Chỉ trích xuất khi có trong input hoặc thật sự cần thiết để kiểm chứng claim.

Cấu trúc:

```json
{
  "time": ["15/3/2024"],
  "location": ["Thái Bình"],
  "source_type": ["thông báo công an", "báo chí chính thống"]
}
```

#### `constraints.time`

Ngày, tháng, năm, giai đoạn, mốc thời gian. Dùng để tránh truy hồi nhầm sự kiện cùng chủ đề nhưng khác thời điểm.

#### `constraints.location`

Địa điểm cần khớp với bằng chứng.

#### `constraints.source_type`

Loại nguồn nên tìm, ví dụ: báo chí, thông báo cơ quan nhà nước, bài đăng mạng xã hội, văn bản pháp lý. Nếu claim không yêu cầu loại nguồn cụ thể thì để `[]`.

### `context_summary`

Tóm tắt ngắn về điều cần kiểm chứng. Trường này giúp người đọc và các bước sau hiểu nhanh bài toán.

Không nên đưa ra kết luận đúng sai.

### `ambiguity_notes`

Danh sách điểm mơ hồ, thiếu thông tin, dễ gây nhầm lẫn hoặc cần làm rõ.

Ví dụ:

```json
"ambiguity_notes": [
  "Claim không nêu ngày xảy ra vụ việc.",
  "Ảnh không đủ rõ để đọc toàn bộ văn bản trên bảng."
]
```

Dùng cho:

- Giải thích vì sao retrieval khó.
- Ưu tiên tìm thêm bằng chứng.
- Đánh giá lỗi input.

### `verification_targets`

Danh sách những điều cần tìm bằng chứng ở bước sau. Đây là checklist cho RAG và fact-checking.

Ví dụ:

```json
"verification_targets": [
  "Tìm nguồn xác nhận Công an tỉnh Thái Bình có khởi tố vụ việc.",
  "Xác minh số lượng đối tượng bị khởi tố là 10.",
  "Xác minh vụ việc liên quan đến lừa đảo hỗ trợ vay vốn online."
]
```

Trường này nên được dùng để đánh giá retrieved context: nếu context không trả lời được các target này, retrieval chưa đủ.

## Cách dùng output để test RAG

### Test một query duy nhất

Dùng `primary_retrieval_query` làm query chính. So sánh kết quả với raw claim và `normalized_claim`.

### Test theo từng mệnh đề

Chạy retrieval trên từng `claim_atoms[].retrieval_queries`. Cách này tốt hơn cho claim dài vì mỗi atom có thể cần bằng chứng khác nhau.

### Test hybrid retrieval

Kết hợp:

- Dense query: `search_queries.semantic`
- Sparse query: `search_queries.keywords`
- Entity boost: `key_entities`
- Constraint filter: `constraints`

### Test multimodal retrieval

Nếu `retrieval_focus.image` hoặc `retrieval_focus.cross_modal` là `true`, dùng:

- `visual_observations[].text`
- `visual_observations[].visible_evidence`
- `search_queries.visual`

để tìm ảnh, caption hoặc đoạn văn liên quan đến nội dung hình ảnh.

### Đánh giá kết quả retrieval

Một retrieved context tốt nên đáp ứng được phần lớn `verification_targets` và các `claim_atoms` có `priority` là `high`.

Nên ghi log:

- Query đã dùng.
- Atom nào được cover.
- Verification target nào được cover.
- Constraint nào bị vi phạm.
- Kết quả có bị nhầm entity, nhầm thời gian, nhầm địa điểm hay không.

## Khuyến nghị cho dataset test

- Lưu raw claim, normalized claim, primary query và atom queries riêng biệt.
- Đánh giá Recall@K theo từng atom thay vì chỉ theo claim.
- Tách lỗi refine và lỗi retrieval: refine sai atom/entity thì không nên tính là lỗi retrieval.
- Với claim có ảnh, đánh giá riêng text retrieval, image retrieval và cross-modal retrieval.
- Với claim thiếu thời gian hoặc địa điểm, kiểm tra `ambiguity_notes` để biết kết quả retrieval có thể mơ hồ.
