import re


import re

def output_split(generated_text):
    # 找到起始点的文本
    marker = "output must be valid JSON.Your final output should only have one result.And you should start with 'final output' before final output"

    # 提取 marker 后的内容
    marker_index = generated_text.find(marker)
    if marker_index != -1:
        # 只分析 marker 后的部分
        generated_text = generated_text[marker_index + len(marker):]
        # print(generated_text)
    else:
        # 如果没有找到 marker，返回 None
        return None, None, None, None

    def extract_first_match(pattern, text):
        """
        使用正则表达式提取第一个匹配项。
        """
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    # 正则模式
    patterns = {
        "has_issue": r"['\"]has\\?_issue['\"]:\s*(true|false|null|['\"](.*?)['\"])",
        "bug_type": r"['\"]bugtype\\?_out['\"]:\s*['\"](.*?)['\"]",
        "defect_code": r"['\"]defect\\?_code['\"]:\s*([^\n]+)",
        "fix": r"['\"]fix['\"]:\s*([^\n]+)"
    }
    def clean_value(value):
        """
        对提取的值进行清理：
        1. 删除第一个字符。
        2. 如果最后一个字符是逗号，则删除最后两个字符。
        3. 否则，删除最后一个字符。
        """
        if value:
            # 如果第一个字符是单引号或双引号，则删除第一个字符
            if value[0] in ("'", '"'):
                value = value[1:]
            # 删除最后一个或两个字符
            if value.endswith(','):
                value = value[:-2]
            elif value.endswith(("'", '"')):
                value = value[:-1]
            # 去除可能的多余空白字符
            return value.strip()
        return value
    def truncate_before_fix(value):
        """
        如果 value 中包含 "fix" 或 'fix'，则截断到 "fix" 或 'fix' 前边。
        """
        if not value:
            return value

        # 使用正则表达式查找 "fix" 或 'fix'，忽略大小写
        match = re.search(r'["\']?fix["\']?', value, re.IGNORECASE)
        if match:
            # 截断到 "fix" 或 'fix' 前边
            return value[:match.start()].strip()
        return value
    def truncate_before_similar(value):
        """
        如果 value 中包含 "fix" 或 'fix'，则截断到 "fix" 或 'fix' 前边。
        """
        if not value:
            return value

        # 使用正则表达式查找 "fix" 或 'fix'，忽略大小写
        match = re.search(r'["\']?similarFix["\']?', value, re.IGNORECASE)
        if match:
            # 截断到 "fix" 或 'fix' 前边
            return value[:match.start()].strip()
        return value
    # 提取字段
    hasissue_value = extract_first_match(patterns["has_issue"], generated_text)
    bug_type_value = extract_first_match(patterns["bug_type"], generated_text)
    defect_code_value = extract_first_match(patterns["defect_code"], generated_text)
    fix_value = extract_first_match(patterns["fix"], generated_text)
    # 清理提取的值
    defect_code_value = clean_value(defect_code_value)
    fix_value = clean_value(fix_value)
    # 防止出现输出都在同一行的情况
    defect_code_value = truncate_before_fix(defect_code_value)
    defect_code_value = clean_value(defect_code_value)
    fix_value = truncate_before_similar(fix_value)
    fix_value = clean_value(fix_value)

    # print(hasissue_value)
    # print(bug_type_value)
    print(defect_code_value)
    print(fix_value)

    return hasissue_value, bug_type_value, defect_code_value, fix_value


text = """
        You are a coding assistant. Analyze the old_code I gave you, and use similarFix as a reference.
            - First, identify the defect_code in old_code by analyzing it in comparison with the similarFix examples.
            - Second, generate a fix for the identified defect_code.
            - Third, classify the defect into one of the predefined bug types.
        Input:
                old_code: ACRA.log.e(LOG_TAG,"ACRA is disabled for " + context.getPackageName() + " - forwarding uncaught Exception on to default ExceptionHandler")
                similarFix: [{'old_code': '\n   }\n \n   @Override\n  public List<INPUT> getSortedDependenciesOf(List<INPUT> roots) {\n\n     return getDependenciesOf(roots, true);\n   }\n \n', 'defect_code': 'List<INPUT>', 'bugType': 'CHANGE_IDENTIFIER', 'fix_code': 'ImmutableList<INPUT>'}, {'old_code': '\n   }\n \n   /**\n\n    * Put all regions under /hbase/replication/regions znode will lead to too many children because\n   * of the huge number of regions in real production environment. So here we use hash of encoded\n   * region name to distribute the znode into multiple znodes. <br>\n\n\n\n\n    * So the final znode path will be format like this:\n    *\n    * <pre>\n   * /hbase/replication/regions/e1/ff/dd04e76a6966d4ffa908ed0586764767-100\n\n    * </pre>\n    *\n   * The e1 indicate the first level hash of encoded region name, and the ff indicate the second\n   * level hash of encoded region name, the 100 indicate the peer id. <br>\n   * Note that here we use two-level hash because if only one-level hash (such as mod 65535), it\n   * will still lead to too many children under the /hbase/replication/regions znode.\n\n\n\n\n\n    * @param encodedRegionName the encoded region name.\n    * @param peerId peer id for replication.\n    * @return ZNode path to persist the max sequence id that we\'ve pushed for the given region and\n    *         peer.\n    */\n   @VisibleForTesting\n  public String getSerialReplicationRegionPeerNode(String encodedRegionName, String peerId) {\n\n     if (encodedRegionName == null || encodedRegionName.length() != RegionInfo.MD5_HEX_LENGTH) {\n       throw new IllegalArgumentException(\n           "Invalid encoded region name: " + encodedRegionName + ", length should be 32.");\n', 'defect_code': '1', 'bugType': 'CHANGE_MODIFIER', 'fix_code': '0'}]{'old_code': 'ACRA.log.e(LOG_TAG,"ACRA is disabled for " + context.getPackageName() + " - forwarding uncaught Exception on to default ExceptionHandler")', 'similarFix': [{'old_code': '\n   }\n \n   @Override\n  public List<INPUT> getSortedDependenciesOf(List<INPUT> roots) {\n\n     return getDependenciesOf(roots, true);\n   }\n \n', 'defect_code': 'List<INPUT>', 'bugType': 'CHANGE_IDENTIFIER', 'fix_code': 'ImmutableList<INPUT>'}, {'old_code': '\n   }\n \n   /**\n\n    * Put all regions under /hbase/replication/regions znode will lead to too many children because\n   * of the huge number of regions in real production environment. So here we use hash of encoded\n   * region name to distribute the znode into multiple znodes. <br>\n\n\n\n\n    * So the final znode path will be format like this:\n    *\n    * <pre>\n   * /hbase/replication/regions/e1/ff/dd04e76a6966d4ffa908ed0586764767-100\n\n    * </pre>\n    *\n   * The e1 indicate the first level hash of encoded region name, and the ff indicate the second\n   * level hash of encoded region name, the 100 indicate the peer id. <br>\n   * Note that here we use two-level hash because if only one-level hash (such as mod 65535), it\n   * will still lead to too many children under the /hbase/replication/regions znode.\n\n\n\n\n\n    * @param encodedRegionName the encoded region name.\n    * @param peerId peer id for replication.\n    * @return ZNode path to persist the max sequence id that we\'ve pushed for the given region and\n    *         peer.\n    */\n   @VisibleForTesting\n  public String getSerialReplicationRegionPeerNode(String encodedRegionName, String peerId) {\n\n     if (encodedRegionName == null || encodedRegionName.length() != RegionInfo.MD5_HEX_LENGTH) {\n       throw new IllegalArgumentException(\n           "Invalid encoded region name: " + encodedRegionName + ", length should be 32.");\n', 'defect_code': '1', 'bugType': 'CHANGE_MODIFIER', 'fix_code': '0'}]}
        Standard output Example:{'has_issue': 'True', 'bugtype_out': 'CHANGE_CALLER_IN_FUNCTION_CALL', 'defect_code': "stacktrace.indexOf(':')", 'fix': "firstLine.indexOf(':')"}defect_code is the defective code you found from sourceBeforeFix, and fix is the repair code you gave for the identified defect_code. Please note that this is just a template, and the code and other information inside is just an example, not what you need to refer to.
        output must be valid JSON.Your final output should only have one result.And you should start with 'final output' before final output
        
        final output:
        {
            "has\_issue": "True",
            "bugtype\_out": "CHANGE\_MODIFIER",
            "defect\_code": "1",
            "fix": "ImmutableList<INPUT>",
            "description": "List<INPUT> should be changed to ImmutableList<INPUT> in getSortedDependenciesOf function"
        }
"""
output_split(text)