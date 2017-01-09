function evaluate_multi_label(predict, target)
  if not predict:isSameSizeAs(target) then
    print("output size must match target size!")
    os.exit(0)
  end
  local predict_of_target = torch.cmul(predict, target:typeAs(predict))
  local number_of_target = torch.sum(target:type("torch.LongTensor"), 2)
  local predict_sorted, _ = torch.sort(predict, true)
  local topk_one_hot = torch.zeros(predict:size()):scatter(2, number_of_target, 1):type("torch.ByteTensor")
  local topk_expand = torch.expandAs(predict_sorted:maskedSelect(topk_one_hot):view(-1,1), predict_sorted)
  local predict_labels = torch.ge(predict_of_target, topk_expand)
  local number_of_true = torch.sum(predict_labels, 2)
  local number_of_all = torch.sum(torch.ge(predict, topk_expand), 2)
  local mean = torch.mean(torch.cdiv(number_of_true:type("torch.DoubleTensor"),number_of_all:type("torch.DoubleTensor")))
  return mean
end

output = torch.Tensor({{0.2,0.3,0.5},{0.5,0.3,0.4}})
target = torch.Tensor({{0,0,1}, {1,0,1}})
print (evaluate_multi_label(output,target))
